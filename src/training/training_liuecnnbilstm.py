import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from src.utils.helper_fns import (
    log_per_class_f1, extract_beat_aligned_sequences, z_normalize,
    load_ecg_data, filter_ecg, get_dataloaders, train_epoch,
    validate, write_hparams, compute_class_weights,
    split_train_val_by_patient, get_exclude_sequence_ids,
    evaluate_macro_f1, save_model_with_metadata, build_model
)
from src.models.liu_cnn_bilstm import LiuCNNBilstm
from src.utils.scheduler import WarmupCosineScheduler

def main(results_path="./res"):
    config_path = "conf/liuecnnbilstm.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    config["SEQ_LEN"] = int(config["Fs"] * config["seq_dur"])

    # === Load and preprocess data ===
    x_raw, y_raw, patient_id, lead, *_ = load_ecg_data(config)
    x_filtered = filter_ecg(x_raw, config)

    x_seq, y_seq, meta_seq = extract_beat_aligned_sequences(
        ecg_signal=x_filtered, label_signal=y_raw, fs=config["Fs"],
        beats_per_seq=config["heart_beats"], seq_len_seconds=config["seq_dur"],
        patient_signal=patient_id, lead_signal=lead, beat_aligned=False
    )

    exclude_ids = get_exclude_sequence_ids(config)
    keep_mask = ~np.isin(meta_seq[:, 2], exclude_ids)
    x_seq, y_seq, meta_seq = x_seq[keep_mask], y_seq[keep_mask], meta_seq[keep_mask]
    x_seq = np.array([z_normalize(seq) for seq in x_seq])

    x_train, y_train, x_val, y_val = split_train_val_by_patient(
        x_seq, y_seq, meta_seq, train_fraction=0.8, random_seed=42
    )

    print("\n--- Dataset Summary ---")
    print(f"Train patients: {len(np.unique(meta_seq[:, 0]))}")
    print(f"Train sequences: {x_train.shape[0]}")
    print(f"Validation sequences: {x_val.shape[0]}")

    class_weights = compute_class_weights(y_train, config["num_classes"])
    class_weights[0] *= 2.0
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda()
    print(f"Class weights: {class_weights_tensor}")

    train_loader, val_loader = get_dataloaders(
        x_train, y_train, x_val, y_val, config["batch_size"]
    )

    # === Build model ===
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["initial_lr"])

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=config["epochs"],
        min_lr=1e-5
    )

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100,
        weight=class_weights_tensor,
        label_smoothing=config["label_smoothing"]
    ).cuda()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    # Use model name and timestamp to generate run-specific log folder
    run_name = config["name"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(base_dir, "logs", "runs", f"{run_name}_{timestamp}")

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    best_macro_f1 = 0.0
    epochs_no_improve = 0
    val_loss = 0.0
    best_model_name = None

    try:
        for epoch in range(config["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}")
            writer.add_scalar("Loss/train", train_loss, epoch)

            val_loss = validate(model, val_loader, loss_fn)
            print(f"                   Val Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/val", val_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)
            print(f"Current LR: {current_lr:.6f}")

            macro_f1 = evaluate_macro_f1(model, val_loader)
            writer.add_scalar("MacroF1/val", macro_f1, epoch)
            print(f"                   Macro F1: {macro_f1:.4f}")

            if (epoch + 1) % config["f1_eval_interval"] == 0 or epoch == config["epochs"] - 1:
                _ = log_per_class_f1(model, val_loader, epoch, writer, config["target_names"])

            scheduler.step()

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
                print("Early stopping triggered based on Macro F1.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully...\n")

    finally:
        writer.close()
        writer = SummaryWriter(log_dir)
        write_hparams(writer, config, val_loss=val_loss, macrof1=best_macro_f1)
        writer.close()

        if best_model_name is not None:
            model.load_state_dict(torch.load(best_model_name))
        else:
            print("No best model saved. Skipping final model load.")

if __name__ == "__main__":
    main()
