import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import json
from bixecg.utils.scheduler import WarmupCosineScheduler

from bixecg.utils.helper_fns import (
    log_per_class_f1, extract_beat_aligned_sequences,
    z_normalize, load_ecg_data, filter_ecg,
    get_dataloaders, train_epoch, validate, write_hparams,
    build_model, compute_class_weights, get_device,
    load_ecg_data, save_model_with_metadata,
    split_train_val_by_patient, get_exclude_sequence_ids,
    evaluate_macro_f1)

def main(results_path="./res"):
    config = json.load(open("conf/jimenezcnn1d.json", "r"))
    # pretty print the config
    print(json.dumps(config, indent=4))
    config["SEQ_LEN"] = int(config["Fs"] * config["seq_dur"])

    device = get_device(config)

    # Data Loading and Preprocessing
    x_train_raw, y_train_raw, patient_id_train, lead_train, _, _, _, _ = load_ecg_data(config)
    x_train_filtered = filter_ecg(x_train_raw, config)

    x_train_seq, y_train_seq, train_meta_seq = extract_beat_aligned_sequences(
        ecg_signal=x_train_filtered, label_signal=y_train_raw, fs=config["Fs"],
        beats_per_seq=config["heart_beats"], seq_len_seconds=config["seq_dur"],
        patient_signal=patient_id_train, lead_signal=lead_train, beat_aligned=False
    )
    
    exclude_ids = get_exclude_sequence_ids(config)

    # Filtering
    train_keep_mask = ~np.isin(train_meta_seq[:, 2], exclude_ids)

    # Apply mask
    x_train_seq = x_train_seq[train_keep_mask]
    y_train_seq = y_train_seq[train_keep_mask]
    train_meta_seq = train_meta_seq[train_keep_mask]

    # Normalize
    x_train_seq = np.array([z_normalize(seq) for seq in x_train_seq])

    x_train_seq, y_train_seq, x_val_seq, y_val_seq = split_train_val_by_patient(
        x_seq=x_train_seq,
        y_seq=y_train_seq,
        meta_seq=train_meta_seq,
        train_fraction=0.8,
        random_seed=42
    )

    # Print number of patients and their IDs
    print("\n--- Dataset Summary ---")
    print(f"Train patients: {len(np.unique(train_meta_seq[:, 0]))} ({np.unique(train_meta_seq[:, 0]).tolist()})")

    # Print number of sequences
    print(f"Train sequences: {x_train_seq.shape[0]}")
    print(f"Validation sequences: {x_val_seq.shape[0]}")

    # Class weights
    class_weights = compute_class_weights(y_train_seq, config["model"]["params"]["num_classes"])
    class_weights[0] *= 2.0  # Adjust class weights for No Wave class
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"Class weights: {class_weights_tensor}")

    # Data Loaders
    train_loader, val_loader = get_dataloaders(x_train_seq, y_train_seq, x_val_seq, y_val_seq, config["batch_size"])

    # Model
    model = build_model(config).to(device)
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["initial_lr"])

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=2,         # for example: 5 epochs warmup
        total_epochs=config["epochs"],  # total training epochs
        min_lr=1e-5              # minimum learning rate after decay
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100, weight=class_weights_tensor, label_smoothing=0.05
    ).to(device)

    # TensorBoard logging setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Use model name and timestamp to generate run-specific log folder
    run_name = config["name"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(base_dir, "logs", "runs", f"{run_name}_{timestamp}")

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    epochs_no_improve = 0
    macrof1 = 0.0

    best_macro_f1 = 0.0 #asd
    epochs_no_improve = 0

    val_loss = 0.0 
    best_model_name = None

    try:
        for epoch in range(config["epochs"]):
            # === Training ===
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}")
            writer.add_scalar("Loss/train", train_loss, epoch)

            # === Validation ===
            val_loss = validate(model, val_loader, loss_fn)
            print(f"                   Val Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/val", val_loss, epoch)

            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)
            print(f"Current LR: {current_lr:.6f}")

            # === F1 evaluation every epoch ===
            macro_f1 = evaluate_macro_f1(model, val_loader)

            writer.add_scalar("MacroF1/val", macro_f1, epoch)
            print(f"                   Macro F1: {macro_f1:.4f}")

            # === Full per-class F1 logging only every N epochs ===
            if (epoch + 1) % config["f1_eval_interval"] == 0 or epoch == config["epochs"] - 1:
                _ = log_per_class_f1(
                    model, val_loader, epoch, writer, config["target_names"]
                )

            # Scheduler step
            scheduler.step()

            # === Early stopping based on macro F1 ===
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                epochs_no_improve = 0
                best_model_name = save_model_with_metadata(
                        model=model,
                        config=config,
                        results_path=results_path,
                        best_macro_f1=best_macro_f1,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        current_lr=current_lr
                )            
            else:
                epochs_no_improve += 1
                print(f"                   No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= config["patience"]:
                print("Early stopping triggered based on Macro F1!")
                break
    except KeyboardInterrupt:
        print("\n\n=== KeyboardInterrupt detected! Gracefully exiting... ===\n")

    finally:
        # Final HPs with val metrics
        writer.close()

        writer = SummaryWriter(log_dir)
        write_hparams(writer, config, val_loss=val_loss, macrof1=macrof1)
        writer.close()

        # Load the best model for evaluation
        if best_model_name is not None:
            model.load_state_dict(torch.load(best_model_name))
        else:
            print("No best model saved. Skipping final model load.")

        writer = SummaryWriter(log_dir)

        writer.close()

if __name__ == "__main__":
    main()