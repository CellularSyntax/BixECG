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
from bixecg.models.models import get_model
import traceback
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import json
import os
import warnings
import random

import time

from bixecg.utils.scheduler import WarmupCosineScheduler
from bixecg.utils.helper_fns import (
    log_per_class_f1, extract_beat_aligned_sequences,
    z_normalize, load_ecg_data, filter_ecg,
    get_dataloaders, train_epoch, validate, write_hparams,
    build_model, compute_class_weights,
    load_ecg_data, redirect_output_to_file,
    split_train_val_by_patient, get_device,
    get_exclude_sequence_ids, evaluate_macro_f1)


amp_precision = torch.bfloat16
weight_precision = torch.float32
enable_mixed_precision = False

# enable_mixed_precision = True
# 1128/1128 [11:42<00:00,  1.61it/s, loss=0.394
#
# enable_mixed_precision = False
# 1128/1128 [20:09<00:00,  1.07s/it, loss=0.391

torch.autograd.set_detect_anomaly(False)


seed = 42

class BaseHPO:
    def __init__(self):
        pass

    def objective(self, trial: optuna.Trial, config_path: str, results_path: str):
        raise NotImplementedError("This method should be overridden by subclasses") 
    
    
    def get_run_name(self, config: dict) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    
    def init_and_run(self, trial, config, results_path):
        device = get_device(config)
        
        

        batch_size = config["batch_size"]
        writer = None

        while(batch_size > 8):
            try:
                print("#############################")
                print(config)
                print("#############################")

                time.sleep(1)
                torch.cuda.empty_cache()


                # TensorBoard logging setup
                run_name = self.get_run_name(config)
                
                start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"bs{config["batch_size"]}_{run_name}_{start_timestamp}"
                run_name = run_name.replace(" ", "")
                log_dir = os.path.join(
                    "./logs/runs", run_name + "_" +
                    start_timestamp
                )

                print(f"run dir: {run_name}")
                writer = SummaryWriter(log_dir)

                # Data Loading and Preprocessing
                x_train_raw, y_train_raw, patient_id_train, lead_train, _, _, _, _ = load_ecg_data(config)
                x_train_filtered = filter_ecg(x_train_raw, config)
                
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
                print(f"normalized min = {np.min(x_train_seq)}, max = {np.max(x_train_seq)}")

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

                print(f"Class weights: {class_weights}")

                # Data Loaders
                train_loader, val_loader = get_dataloaders(x_train_seq, y_train_seq, x_val_seq, y_val_seq, batch_size)

                # Model
                model = build_model(config).to(device)
                
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

                return self.run(
                    trial, 
                    config,
                    train_loader, 
                    val_loader,
                    model,
                    loss_fn,
                    optimizer,
                    scheduler,
                    results_path,
                    run_name,
                    writer
                )

            except Exception as e:
                stacktrace = str(traceback.format_exc())
                with open(os.path.join(results_path, run_name, "exception.txt"), "w") as f:
                    f.write(str(e))
                    f.write("\n")
                    f.write("\n")
                    f.write(stacktrace)
                if any(err in stacktrace for err in ["out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED"]):
                    # Handle OOM error by reducing batch_size
                    old_batch_size = batch_size
                    batch_size = int(batch_size * config["batch_size_reduction_factor"])
                    config["batch_size"] = batch_size
                    print(f"❗️ OutOfMemoryError - retrying with a new batchsize of {batch_size} (was {old_batch_size})")
                    #print(str(traceback.format_exc()))
                else:
                    raise e
            finally:
                model = None
                if writer is not None:
                    writer.close()
    
        raise torch.OutOfMemoryError(f"Can't run training with a batchsize of {batch_size} and the current network configuration")



    def run(self,
            trial, 
            config,
            train_loader, 
            val_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            results_path,
            run_name,
            writer):


        device = get_device(config)

        epochs_no_improve = 0
        macro_f1 = 0.0

        best_macro_f1 = 0.0 #asd
        epochs_no_improve = 0

        val_loss = 0.0 
        best_model_path = None
        
        

        model_dir = os.path.join(results_path, run_name)
        os.makedirs(model_dir, exist_ok=True)

        batch_x, batch_y = next(iter(train_loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        input_size = batch_x.size()
        model_summary = summary(model, verbose=1, input_size=input_size, col_names=["input_size", "output_size", "mult_adds", "num_params"], device=device)
        with open(f"{model_dir}/model_summary.txt", "w") as f:
            f.write(str(model_summary))
        with open(f"{model_dir}/model_summary_single_batch.txt", "w") as f:
            f.write(str(summary(model, verbose=0, input_size=(1, *input_size[1:]), col_names=["input_size", "output_size", "mult_adds", "num_params"])))
        batch_x, batch_y = None, None


        # do_cleanup = True

        if enable_mixed_precision:
            model = model.to(dtype=weight_precision)
        device = get_device(config)

        stop_reason = "N/A"

        try:
            #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for epoch in range(config["epochs"]):
                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("LR", current_lr, epoch)
                print(f"Current LR: {current_lr:.6f}")

                train_loss = None
                # === Training ===
                # with torch.autocast(
                #     device_type=device,
                #     dtype=amp_precision,
                #     enabled=enable_mixed_precision,
                # ):
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
                    with open(os.path.join(model_dir, f"NAN_epoch{epoch+1}_lr{current_lr}.json"), "w") as f:
                        json.dump(metadata, f, indent=4)

                    return train_loss
                writer.add_scalar("Loss/train", train_loss, epoch)

                # === Validation ===
                val_loss = None
                # with torch.autocast(
                #     device_type=device,
                #     dtype=amp_precision,
                #     enabled=enable_mixed_precision,
                # ):
                val_loss = validate(model, val_loader, loss_fn)
                print(f"                   Val Loss: {val_loss:.4f}")
                writer.add_scalar("Loss/val", val_loss, epoch)

                # === F1 evaluation every epoch ===
                macro_f1 = None
                # with torch.autocast(
                #     device_type=device,
                #     dtype=amp_precision,
                #     enabled=enable_mixed_precision,
                # ):
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
                    raise optuna.exceptions.TrialPruned("Trial pruned")

                # === Early stopping based on macro F1 ===
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    epochs_no_improve = 0
                    os.makedirs(model_dir, exist_ok=True)
                    model_filename = f"macroF1_{best_macro_f1:.4f}_epoch{epoch+1}"
                    best_model_path = os.path.join(model_dir, f"{model_filename}.pt")
                    torch.save(model.state_dict(), best_model_path)
                    metadata = {
                        "epoch": epoch + 1,
                        "macro_f1": float(best_macro_f1),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "learning_rate": float(current_lr),
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
                    stop_reason = "Early stopping"
                    # TODO: raise optuna.exceptions.TrialPruned() ?
                    return best_macro_f1

            stop_reason = "Max epochs reached"
            return best_macro_f1
        except KeyboardInterrupt:
            print("\n\n=== KeyboardInterrupt detected! Gracefully exiting... ===\n")
            stop_reason = "KeyboardInterrupt"
            if macro_f1 == 0:
                # not even a single epoch was completed
                trial.set_user_attr("reason", "keyboard interrupt")
            return best_macro_f1
        except Exception as e:
            stop_reason = f"Exception ({str(e)})"
            raise e

        finally:
            # Final HPs with val metrics
            write_hparams(writer, config, val_loss=val_loss, macrof1=macro_f1)
            
            summary_csv = os.path.join(results_path, f"summary.csv")
            sep = ";"
            fw_pass_time = 0.0
            gpu_name = "N/A"
            try:
                x_single = next(iter(train_loader))[0][0:1].to(device)
                fw_start = time.time()
                model(x_single)
                fw_pass_time = time.time() - fw_start
                gpu_name = torch.cuda.get_device_name()
            except Exception as e:
                pass
            if not os.path.isfile(summary_csv):
                with open(summary_csv, "w") as f:
                    line = sep.join(["best_macrof1", "config", "total_params", "total_params_MB", "stopped_due_to", "fw_time_1sample_batch", "gpu"]) + "\n"
                    f.write(line)
            with open(summary_csv, "a") as f:
                line = f"{best_macro_f1:.4f}{sep}{run_name}{sep}{model_summary.total_params}{sep}{(model_summary.total_param_bytes / 1024 / 1024):.2f}{sep}{stop_reason.replace(';', ',.')}{sep}{fw_pass_time:.3f}{sep}{gpu_name}\n"
                f.write(line)
            # # Load the best model for evaluation
            # model.load_state_dict(torch.load(best_model_path))
            # writer = SummaryWriter(log_dir)
            # writer.close()

    def main(self, config_path):
        config = json.load(open(config_path))

        results_path = os.path.join("results", "hpo2", f'{config["name"]}')
        os.makedirs(results_path, exist_ok=True)

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
            #config["SEQ_LEN"] = int(config["Fs"] * config["seq_dur"])
            #return objective(trial, config, x=x_train_filtered, y=y_train_raw, patient_id=patient_id_train, lead=lead_train)
            return self.objective(trial, config_path, results_path)
        
        study.optimize(objective_wrapper, n_trials=60)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("\n")
        print(study.trials_dataframe())
        print("\n")

        print("Best trial:")
        trial = study.best_trial


        print("  Value: ", trial.value)