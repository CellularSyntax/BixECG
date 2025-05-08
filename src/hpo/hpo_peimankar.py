import optuna
import itertools
import torch
import warnings
import json

from src.hpo.base_hpo import BaseHPO


class PeimankarHPO(BaseHPO):
    def __init__(self):
        super().__init__()
    
    def objective(self, trial: optuna.Trial, config_path: str, results_path: str):
        config = json.load(open(config_path))
        
        kernel_sizes = trial.suggest_categorical('kernel_sizes', [3, 11, 31, 51])
        config["model"]["params"]["kernel_sizes"] = [kernel_sizes] * 3

        lstm_hidden_sizes = trial.suggest_categorical('lstm_hidden_sizes', [250, 200, 100, 50])
        #lstm_hidden_sizes = 250
        config["model"]["params"]["lstm_hidden_sizes"] = [int(lstm_hidden_sizes), int(lstm_hidden_sizes/2)]
        
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        config["model"]["params"]["dropouts"] = [float("{:.2f}".format(dropout))] * 2
        

        config["SEQ_LEN"] = int(config["Fs"] * config["seq_dur"])
        

        # Here goes the training loop
        best_val_f1 = self.init_and_run(trial, config, results_path)
        return best_val_f1

    def get_run_name(self, config: dict) -> str:
        return (f"ks{config["model"]["params"]['kernel_sizes']}_"
                f"hidden{config["model"]["params"]['lstm_hidden_sizes']}_do{config["model"]["params"]['dropouts']}")
        
    