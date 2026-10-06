import datetime
import optuna
from optuna.trial import TrialState
import itertools
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from my_dataset import get_train_val_loaders  # <-- you should implement this
# from my_model import MyModel  # <-- your model definition
# from my_train_utils import train_one_epoch, evaluate_model  # <-- your custom training & eval functions
from bixecg.models.models import get_model
import traceback

from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import json
import os
import warnings

torch.autograd.set_detect_anomaly(True)

from bixecg.utils.scheduler import WarmupCosineScheduler
from bixecg.utils.helper_fns import (
    log_per_class_f1, extract_beat_aligned_sequences,
    z_normalize, load_ecg_data, filter_ecg,
    get_dataloaders, train_epoch, validate, write_hparams,
    build_model, compute_class_weights,
    load_ecg_data, redirect_output_to_file,
    split_train_val_by_patient, get_device,
    get_exclude_sequence_ids, evaluate_macro_f1)


def objective(trial: optuna.Trial, config:dict, results_path):

    kernel_sizes = trial.suggest_categorical('kernel_sizes', [3, 11, 31, 51])
    config["model"]["params"]["kernel_sizes"] = [kernel_sizes] * 3

    lstm_hidden_sizes = trial.suggest_categorical('lstm_hidden_sizes', [250, 200, 100, 50])
    #lstm_hidden_sizes = 250
    config["model"]["params"]["lstm_hidden_sizes"] = [int(lstm_hidden_sizes), int(lstm_hidden_sizes/2)]
    
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    config["model"]["params"]["dropouts"] = [float("{:.2f}".format(dropout))] * 2
    

    config["SEQ_LEN"] = int(config["Fs"] * config["seq_dur"])

    # Here goes the training loop
    best_val_f1 = init_and_run(trial, config, results_path)


    return best_val_f1


def init_and_run(trial, config, results_path):
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

    batch_size = config["batch_size"]

    while(batch_size > 8):
        try:

            

            # Data Loaders
            train_loader, val_loader = get_dataloaders(x_train_seq, y_train_seq, x_val_seq, y_val_seq, batch_size)

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

            return run(
                trial, 
                config,
                train_loader, 
                val_loader,
                model,
                loss_fn,
                optimizer,
                scheduler,
                results_path
            )

        except Exception as e:
            if "OutOfMemoryError" in str(traceback.format_exc()):
                # Handle OOM error by reducing batch_size
                old_batch_size = batch_size
                batch_size = int(batch_size / 2)
                print(f"❗️ OutOfMemoryError - retrying with a new batchsize of {batch_size} (was {old_batch_size})")
                #print(str(traceback.format_exc()))
            else:
                raise e
  
    raise torch.OutOfMemoryError(f"Can't run training with a batchsize of {batch_size}")



def run(trial, 
        config,
        train_loader, 
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        results_path):


    epochs_no_improve = 0
    macrof1 = 0.0

    best_macro_f1 = 0.0 #asd
    epochs_no_improve = 0

    val_loss = 0.0 
    best_model_path = None

    # TensorBoard logging setup
    run_name = (f"ks{config["model"]["params"]['kernel_sizes']}_"
                f"hidden{config["model"]["params"]['lstm_hidden_sizes']}_do{config["model"]["params"]['dropouts']}")
    log_dir = os.path.join(
        f"logs/runs/{config["model"]["name"]}", run_name + "_" +
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    writer = SummaryWriter(log_dir)

    # do_cleanup = True

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

            trial.report(macro_f1, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # === Early stopping based on macro F1 ===
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                epochs_no_improve = 0
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join(results_path, run_name)
                os.makedirs(model_dir, exist_ok=True)
                model_filename = f"macroF1_{best_macro_f1:.4f}_epoch{epoch+1}_{timestamp}"
                best_model_path = os.path.join(model_dir, f"{model_filename}.pt")
                torch.save(model.state_dict(), best_model_path)
                metadata = {
                    "epoch": epoch + 1,
                    "macro_f1": float(best_macro_f1),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "learning_rate": float(current_lr),
                    "timestamp": timestamp,
                    "config": config 
                }
                with open(os.path.join(model_dir, f"{model_filename}.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                

                print(f"Saved new best model: {best_model_path} and metadata")
            else:
                epochs_no_improve += 1
                print(f"                   No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= config["patience"]:
                print("Early stopping triggered based on Macro F1!")
                break

        return best_macro_f1
    except KeyboardInterrupt:
        print("\n\n=== KeyboardInterrupt detected! Gracefully exiting... ===\n")

    # except Exception as e:
    #    if "OutOfMemoryError" in str(traceback.format_exc()):
    #        do_post_train_ = False
    #        raise e

    finally:
        # Final HPs with val metrics
        writer.close()

        writer = SummaryWriter(log_dir)
        write_hparams(writer, config, val_loss=val_loss, macrof1=macrof1)
        writer.close()

        # # Load the best model for evaluation
        # model.load_state_dict(torch.load(best_model_path))
        # writer = SummaryWriter(log_dir)
        # writer.close()


def main(args):
    config = json.load(open("conf/peimankarcnnbilstm.json", "r"))

    results_path = os.path.join("results", "hpo", "peimankarcnnbilstm2")
    # # Load data
    # x_train_raw, y_train_raw, patient_id_train, lead_train, _, _, _, _ = load_ecg_data(config)
    # # Bandpass filter the data
    # x_train_filtered = filter_ecg(x_train_raw, config)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    seed = 13
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=15)
    )

    def objective_wrapper(trial):
        return objective(trial, config, results_path)
    
    study.optimize(objective_wrapper, n_trials=50, timeout=3600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)



# 🎯 Launch study
if __name__ == "__main__":

    main(None)

