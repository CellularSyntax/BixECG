import optuna
import itertools
import torch
import warnings

from src.hpo.base_hpo import BaseHPO


class BiXLSTMHPO(BaseHPO):
    def __init__(self):
        super().__init__()
    
    def objective(self, trial: optuna.Trial, config:dict, results_path: str):
    
        
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

        num_block_slstm_at_choices = num_block_slstm_at_dict.values()

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
        config["model"]["params"]["conv1d_kernel_size"] = trial.suggest_categorical('conv1d_kernel_size', [3, 7, 11, 21, 31, 41, 51, 61, 71])
        #conv1d_kernel_size = trial.suggest_int('conv1d_kernel_size', 11, 71, step=10)

        #seq_dur = config["seq_dur"]
        seq_dur = trial.suggest_float("seq_dur", 0.5, 2.5, step=0.25)
        seq_dur = float("{:.2f}".format(seq_dur))
        config["seq_dur"] = seq_dur

        #config["model"]["params"]["sequence_length"] = trial.suggest_categorical('sequence_length', [100, 200, 300])
        
        config["SEQ_LEN"] = int(config["Fs"] * seq_dur)

        config["model"]["params"]["seq_length"] = int(config["Fs"] * seq_dur)

        

        # Here goes the training loop
        best_val_f1 = self.init_and_run(trial, config, results_path)
        return best_val_f1

    def get_run_name(self, config: dict) -> str:
        return (f"seqlen{config['SEQ_LEN']}_emb{config['model']['params']['embedding_dim']}_ks{config['model']['params']['conv1d_kernel_size']}_"
                    f"blocks{config['model']['params']['num_blocks']}_slstmat{config['model']['params']['slstm_at']}_do{config['model']['params']['dropout']}_lr{config['initial_lr']}")
    