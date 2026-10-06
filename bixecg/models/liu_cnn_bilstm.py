import torch
import torch.nn as nn
import torch.nn.functional as F

class LiuCNNBilstm(nn.Module):
    """
    Lightweight CNN-BiLSTM model for ECG delineation, based on:
    Liu et al., Scientific Reports, 2023.
    """

    def __init__(self,
                 input_dim=1,
                 num_classes=4,
                 cnn_channels=[32, 64, 128],
                 kernel_sizes=[5, 5, 3],
                 pool_sizes=[2, 2, 2],
                 lstm_hidden_size=128,
                 lstm_layers=1,
                 dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv_blocks = nn.Sequential()
        in_channels = input_dim

        for i, (out_channels, k, p) in enumerate(zip(cnn_channels, kernel_sizes, pool_sizes)):
            self.conv_blocks.add_module(f"conv_{i}", nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2))
            self.conv_blocks.add_module(f"bn_{i}", nn.BatchNorm1d(out_channels))
            self.conv_blocks.add_module(f"relu_{i}", nn.ReLU())
            self.conv_blocks.add_module(f"pool_{i}", nn.MaxPool1d(kernel_size=p))
            in_channels = out_channels

        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(2 * lstm_hidden_size, num_classes)

    def forward(self, x):
        # Input shape: (B, T, C)
        B, T, _ = x.shape

        x = x.permute(0, 2, 1)      # → (B, C, T)
        x = self.conv_blocks(x)     # → (B, C_out, T_pooled)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)      # → (B, T_pooled, C_out)
        x, _ = self.bilstm(x)       # → (B, T_pooled, 2*hidden)
        logits = self.fc(x)         # → (B, T_pooled, num_classes)

        # Interpolate back to original T
        logits = F.interpolate(logits.permute(0, 2, 1), size=T, mode="linear", align_corners=True)
        return logits.permute(0, 2, 1)  # → (B, T, num_classes)
