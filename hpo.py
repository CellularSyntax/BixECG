import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from my_dataset import get_train_val_loaders  # <-- you should implement this
from my_model import MyModel  # <-- your model definition
from my_train_utils import train_one_epoch, evaluate_model  # <-- your custom training & eval functions

def objective(trial):
    # 🔍 Suggest hyperparameters
    num_blocks = trial.suggest_int('num_blocks', 2, 6)
    max_positions = list(range(num_blocks))
    possible_slstm_at_options = []
    for k in range(1, min(4, num_blocks+1)):  # e.g. 1 to 3 LSTM layers
        possible_slstm_at_options.extend(itertools.combinations(max_positions, k))
    # Sample from feasible options
    slstm_at = trial.suggest_categorical("slstm_at", possible_slstm_at_options)

    # Now suggest other hyperparameters
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    embedding_dim = trial.suggest_categorical('embedding_dim', [16, 32, 64])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    sequence_length = trial.suggest_categorical('sequence_length', [100, 200, 300])

    # ⚙️ Prepare data
    train_loader, val_loader = get_train_val_loaders(batch_size=batch_size)

    # 🧠 Define model
    model = MyModel(embedding_dim=embedding_dim, dropout=dropout).to(device)

    # ⚙️ Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience = 5
    no_improve_epochs = 0

    for epoch in range(20):  # max epochs
        train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_f1 = evaluate_model(model, val_loader, device)
        trial.report(val_f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                break

    return best_val_f1

# 🎯 Launch study
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    study.optimize(objective, n_trials=50, timeout=3600)

    print("Best trial:")
    print(study.best_trial)
