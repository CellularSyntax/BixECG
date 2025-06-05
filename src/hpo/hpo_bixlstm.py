import optuna
import itertools
import torch
import warnings
import json
import os

from src.hpo.base_hpo import BaseHPO


class BiXLSTMHPO(BaseHPO):
    def __init__(self, multi_objective: bool = True, save_to="results", hpo_name="hpo", seed: int = 42, hpo_mode: bool = True, beat_aligned:bool = True):
        super().__init__(multi_objective, save_to, hpo_name, seed, hpo_mode=hpo_mode, beat_aligned=beat_aligned)
        
    
    def objective(self, trial: optuna.Trial, config_path: str, results_path: str):
        config = json.load(open(config_path))

        lr = trial.suggest_float('initial_lr', 0.003, 0.01, step=0.001)
        config["initial_lr"] = lr
        seq_dur = config["seq_dur"]
        seq_dur = float("{:.2f}".format(seq_dur))
        config["seq_dur"] = seq_dur
        config["SEQ_LEN"] = int(config["Fs"] * seq_dur)
        config["model"]["params"]["conv1d_kernel_size"] = 7

        # Here goes the training loop
        best_val_f1 = self.init_and_run(trial, config, results_path)
        return best_val_f1


        num_block_slstm_at_dict = {}
        #for num_blocks in range(2, 7):
        #for num_blocks in range(1, 5):
        for num_blocks in range(1, 6):
            max_positions = list(range(num_blocks))
            #possible_slstm_at_options = [()]
            possible_slstm_at_options = []
            #for k in range(1, min(4, num_blocks+1)):  # e.g. 1 to 3 sLSTM layers
            #for k in range(0, min(3, num_blocks+1)):  # e.g. 0 to 2 sLSTM layers
            for k in range(0, num_blocks+1):  # e.g. 0 to 2 sLSTM layers
                possible_slstm_at_options.extend(itertools.combinations(max_positions, k))
            possible_slstm_at_options = [*map(list, possible_slstm_at_options)]
            for possible_positions in possible_slstm_at_options:
                key = f"{num_blocks}_{"".join([str(at) for at in possible_positions])}"
                val = [num_blocks, possible_positions]
                num_block_slstm_at_dict[key] = val

        num_block_slstm_at_choices = num_block_slstm_at_dict.values()

        ## list of possible choices must not change -> num_block_slstm_at_choices

        #num_blocks_slstm_at_choice = trial.suggest_categorial("num_blocks_slstm_at", num_block_slstm_at_choices)
        # # https://github.com/optuna/optuna/issues/2341
        # # "Some types such as tuple or dictionary are not recommended because there is no guarantee for compatibility across different storage backends (e.g. MySQL and Redis)."
        # # UserWarning: Choices for a categorical distribution should be a tuple of None, bool, int, float and str for persistent storage but contains [2, 4, 5] which is of type list.
        # # warning can be ignored.
        # with warnings.catch_warnings(action="ignore"):
        #     num_blocks, slstm_at = trial.suggest_categorical("num_blocks_slstm_at", num_block_slstm_at_choices)
        #     config["model"]["params"]["num_blocks"] = num_blocks
        #     config["model"]["params"]["slstm_at"] = slstm_at


        config_id = trial.suggest_categorical("num_blocks_slstm_at", range(len(num_block_slstm_at_choices)))
        num_blocks, slstm_at = list(num_block_slstm_at_choices)[config_id]
        config["model"]["params"]["num_blocks"] = num_blocks
        config["model"]["params"]["slstm_at"] = slstm_at
        
        trial.study.set_user_attr("num_block_slstm_at_mapping", {id: list(num_block_slstm_at_choices)[id] for id in range(len(num_block_slstm_at_choices))})
        trial.set_user_attr("num_blocks", num_blocks)
        trial.set_user_attr("slstm_at", slstm_at)

        trial.set_user_attr("num_blocks_slstm_at", f"{num_blocks}_{str(slstm_at).replace(" ", "")}")
        trial.set_user_attr("block_config", f"{num_blocks}_{''.join('s' if i in slstm_at else 'm' for i in range(num_blocks))}")


        ## the performance also changes with a different batchsize
        ## could use max batch_size suggestion and reduce if needed
        #batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        #config["batch_size"] = batch_size
        
        # Now suggest other hyperparameters
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        dropout = float("{:.2f}".format(dropout))
        config["model"]["params"]["dropout"] = dropout
        #embedding_dim = trial.suggest_categorical('embedding_dim', [12, 16, 20, 24, 28, 32, 36, 40, 48, 64, 80, 100])
        config["model"]["params"]["embedding_dim"] = trial.suggest_int('embedding_dim', 8, 40, step=4)
        #config["model"]["params"]["conv1d_kernel_size"] = trial.suggest_categorical('conv1d_kernel_size', [3, 7, 11, 21, 31, 41, 51, 61, 71])
        config["model"]["params"]["conv1d_kernel_size"] = trial.suggest_categorical('conv1d_kernel_size', [4, 11, 21, 31, 41, 51, 61])
        #conv1d_kernel_size = trial.suggest_int('conv1d_kernel_size', 11, 71, step=10)

        config["model"]["params"]["num_heads"] = trial.suggest_categorical('num_heads', [1, 2, 4])

        seq_dur = config["seq_dur"]
        #seq_dur = trial.suggest_float("seq_dur", 0.5, 2.0, step=0.25)
        seq_dur = float("{:.2f}".format(seq_dur))
        config["seq_dur"] = seq_dur

        #config["model"]["params"]["sequence_length"] = trial.suggest_categorical('sequence_length', [100, 200, 300])
        
        config["SEQ_LEN"] = int(config["Fs"] * seq_dur)

        print(f"seq_dur: {seq_dur}, Fs: {config['Fs']}, SEQ_LEN: {config['SEQ_LEN']}")

        config["model"]["params"]["seq_length"] = int(config["Fs"] * seq_dur)


        config["model"]["params"]["round_slstm_proj_up_dim_up"] = False
        config["model"]["params"]["round_slstm_proj_up_to_multiple_of"] = 1
        config["model"]["params"]["qkv_proj_blocksize"] = 2
        config["model"]["params"]["round_mlstm_proj_up_dim_up"] = True
        config["model"]["params"]["round_mlstm_proj_up_to_multiple_of"] = config["model"]["params"]["num_heads"]

        

        # Here goes the training loop
        best_val_f1 = self.init_and_run(trial, config, results_path)
        return best_val_f1

    def get_run_name(self, config: dict) -> str:
        return (f"seqlen{config['SEQ_LEN']}_emb{config['model']['params']['embedding_dim']}_ks{config['model']['params']['conv1d_kernel_size']}_"
                    f"blocks{config['model']['params']['num_blocks']}_nh{config['model']['params']['num_heads']}_slstmat{config['model']['params']['slstm_at']}_do{config['model']['params']['dropout']}_lr{config['initial_lr']}")
    