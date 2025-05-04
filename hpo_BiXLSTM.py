import datetime
import optuna
from optuna.trial import TrialState
import itertools
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from my_dataset import get_train_val_loaders  # <-- you should implement this
# from my_model import MyModel  # <-- your model definition
# from my_train_utils import train_one_epoch, evaluate_model  # <-- your custom training & eval functions
from src.models.models import get_model
import traceback
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import json
import os
import warnings
import random



from src.utils.scheduler import WarmupCosineScheduler
from src.utils.helper_fns import (
    log_per_class_f1, extract_beat_aligned_sequences,
    z_normalize, load_ecg_data, filter_ecg,
    get_dataloaders, train_epoch, validate, write_hparams,
    build_model, compute_class_weights,
    load_ecg_data, redirect_output_to_file,
    split_train_val_by_patient, get_device,
    get_exclude_sequence_ids, evaluate_macro_f1)


amp_precision = torch.bfloat16
weight_precision = torch.float32
enable_mixed_precision = True

# enable_mixed_precision = True
# 1128/1128 [11:42<00:00,  1.61it/s, loss=0.394
#
# enable_mixed_precision = False
# 1128/1128 [20:09<00:00,  1.07s/it, loss=0.391

torch.autograd.set_detect_anomaly(False)


seed = 42

def objective(trial: optuna.Trial, config:dict, num_block_slstm_at_choices, results_path):
   
    ## list of possible choices must not change -> num_block_slstm_at_choices

    #num_blocks_slstm_at_choice = trial.suggest_categorial("num_blocks_slstm_at", num_block_slstm_at_choices)
    # # https://github.com/optuna/optuna/issues/2341
    # # "Some types such as tuple or dictionary are not recommended because there is no guarantee for compatibility across different storage backends (e.g. MySQL and Redis)."
    # # UserWarning: Choices for a categorical distribution should be a tuple of None, bool, int, float and str for persistent storage but contains [2, 4, 5] which is of type list.
    # # warning can be ignored.
    with warnings.catch_warnings(action="ignore"):
        num_blocks, slstm_at = trial.suggest_categorical("num_blocks_slstm_at", num_block_slstm_at_choices)
        config["model"]["params"]["num_blocks"] = num_blocks
        config["model"]["params"]["slstm_at"] = slstm_at
    

    ## the performance also changes with a different batchsize
    ## could use max batch_size suggestion and reduce if needed
    #batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    #config["batch_size"] = batch_size
    
    # Now suggest other hyperparameters
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    dropout = float("{:.2f}".format(dropout))
    config["model"]["params"]["dropout"] = dropout
    #embedding_dim = trial.suggest_categorical('embedding_dim', [12, 16, 20, 24, 28, 32, 36, 40, 48, 64, 80, 100])
    config["model"]["params"]["embedding_dim"] = trial.suggest_int('embedding_dim', 12, 100, step=4)
    config["model"]["params"]["conv1d_kernel_size"] = trial.suggest_categorical('conv1d_kernel_size', [11, 21, 31, 41, 51, 61, 71])
    #conv1d_kernel_size = trial.suggest_int('conv1d_kernel_size', 11, 71, step=10)

    #seq_dur = config["seq_dur"]
    seq_dur = trial.suggest_float("seq_dur", 1.0, 3.0, step=0.25)
    seq_dur = float("{:.2f}".format(seq_dur))
    config["seq_dur"] = seq_dur

    #config["model"]["params"]["sequence_length"] = trial.suggest_categorical('sequence_length', [100, 200, 300])
    
    config["SEQ_LEN"] = int(config["Fs"] * seq_dur)

    config["model"]["params"]["seq_length"] = int(config["Fs"] * seq_dur)

    print("#############################")
    print(config)
    print("#############################")

    # Here goes the training loop
    best_val_f1 = init_and_run(trial, config, results_path)

    


    return best_val_f1


# def find_max_batch_size(batch_size):
#     while(batch_size > 8):
#         try:
            
    
def init_and_run(trial, config, results_path):
    device = get_device(config)
    
    # Data Loading and Preprocessing
    x_train_raw, y_train_raw, patient_id_train, lead_train, _, _, _, _ = load_ecg_data(config)
    x_train_filtered = filter_ecg(x_train_raw, config)

    batch_size = config["batch_size"]

    while(batch_size > 8):
        try:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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
            train_loader, val_loader = get_dataloaders(x_train_seq, y_train_seq, x_val_seq, y_val_seq, batch_size)

            # Model
            model = build_model(config).to(device)
            batch_x, batch_y = next(iter(train_loader))
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            input_size = batch_x.size()
            summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params"])
            #summary(model)
            batch_x, batch_y = None, None

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
                config["batch_size"] = batch_size
                print(f"❗️ OutOfMemoryError - retrying with a new batchsize of {batch_size} (was {old_batch_size})")
                #print(str(traceback.format_exc()))
            else:
                raise e
  
    raise torch.OutOfMemoryError(f"Can't run training with a batchsize of {batch_size} and the current network configuration")



def run(trial, 
        config,
        train_loader, 
        val_loader,
        model,
        loss_fn,
        optimizer,
        scheduler,
        results_path="./results"):


    epochs_no_improve = 0
    macrof1 = 0.0

    best_macro_f1 = 0.0 #asd
    epochs_no_improve = 0

    val_loss = 0.0 
    best_model_path = None
    
    # TensorBoard logging setup
    run_name = (f"seqlen{config['SEQ_LEN']}_emb{config['model']['params']['embedding_dim']}_ks{config['model']['params']['conv1d_kernel_size']}_"
                f"blocks{config['model']['params']['num_blocks']}_slstmat{config['model']['params']['slstm_at']}_do{config['model']['params']['dropout']}")
    run_name = run_name.replace(" ", "")
    log_dir = os.path.join(
        "./logs/runs", run_name + "_" +
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    writer = SummaryWriter(log_dir)

    # do_cleanup = True

    if enable_mixed_precision:
        model = model.to(dtype=weight_precision)
    device = get_device(config)

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(results_path, run_name)
        for epoch in range(config["epochs"]):
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)
            print(f"Current LR: {current_lr:.6f}")

            train_loss = None
            # === Training ===
            with torch.autocast(
                device_type=device,
                dtype=amp_precision,
                enabled=enable_mixed_precision,
            ):
                train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}")
            if math.isnan(train_loss):
                trial.report(train_loss, epoch)
                os.makedirs(model_dir, exist_ok=True)
                metadata = {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "learning_rate": float(current_lr),
                    "config": config 
                }
                with open(os.path.join(model_dir, f"NAN_epoch{epoch+1}_lr{current_lr}_{timestamp}.json"), "w") as f:
                    json.dump(metadata, f, indent=4)

                return train_loss
            writer.add_scalar("Loss/train", train_loss, epoch)

            # === Validation ===
            val_loss = None
            with torch.autocast(
                device_type=device,
                dtype=amp_precision,
                enabled=enable_mixed_precision,
            ):
                val_loss = validate(model, val_loader, loss_fn)
            print(f"                   Val Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/val", val_loss, epoch)

            # === F1 evaluation every epoch ===
            macro_f1 = None
            with torch.autocast(
                device_type=device,
                dtype=amp_precision,
                enabled=enable_mixed_precision,
            ):
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
    config = json.load(open("conf/BiXLSTM1.json", "r"))

    results_path = os.path.join("results", "hpo", config["name"])

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # Load data
    # x_train_raw, y_train_raw, patient_id_train, lead_train, _, _, _, _ = load_ecg_data(config)
    # # Bandpass filter the data
    # x_train_filtered = filter_ecg(x_train_raw, config)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        sampler=sampler,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    def objective_wrapper(trial):

        num_block_slstm_at_dict = {}
        for num_blocks in range(2, 7):
            max_positions = list(range(num_blocks))
            #possible_slstm_at_options = [()]
            possible_slstm_at_options = []
            for k in range(1, min(4, num_blocks+1)):  # e.g. 1 to 3 sLSTM layers
                possible_slstm_at_options.extend(itertools.combinations(max_positions, k))
            possible_slstm_at_options = [*map(list, possible_slstm_at_options)]
            for possible_positions in possible_slstm_at_options:
                key = f"{num_blocks}_{"".join([str(at) for at in possible_positions])}"
                val = [num_blocks, possible_positions]
                num_block_slstm_at_dict[key] = val
        
        #config["SEQ_LEN"] = int(config["Fs"] * config["seq_dur"])
        #return objective(trial, config, x=x_train_filtered, y=y_train_raw, patient_id=patient_id_train, lead=lead_train)
        return objective(trial, config, num_block_slstm_at_dict.values(), results_path)
    
    study.optimize(objective_wrapper, n_trials=30, timeout=3600)

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

