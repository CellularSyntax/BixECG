import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

class DropHiddenState(nn.Module):
    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        return x[0]


class PeimankarCnnBilstm(nn.Module):
    """
    CNN-BiLSTM model for ECG delineation based on:
    Peimankar et al., 2020 (Expert Systems with Applications).
    """
    def __init__(self,
                 input_dim=1,
                 num_classes=4,
                 cnn_channels=[32, 64, 128],
                 kernel_sizes=[3, 3, 3],
                 lstm_hidden_sizes=[250, 125],
                 lstm_layers=[1, 1],
                 bidirectional=True,
                 dropouts=[0.3, 0.3]):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv_blocks = nn.Sequential()
        in_channels = input_dim

        for i, (out_channels, k) in enumerate(zip(cnn_channels, kernel_sizes)):
            self.conv_blocks.add_module(f"conv_{i}", nn.Conv1d(in_channels, out_channels, kernel_size=k))
            self.conv_blocks.add_module(f"relu_{i}", nn.ReLU())
            self.conv_blocks.add_module(f"zp_{i}", nn.ZeroPad1d(1))  # zero-padding to help maintain length
            in_channels = out_channels

        self.lstm_blocks = nn.Sequential()
        input_size = cnn_channels[-1]
        factor = 2 if bidirectional else 1

        for i, (hidden_size, num_layers, dropout) in enumerate(zip(lstm_hidden_sizes, lstm_layers, dropouts)):
            self.lstm_blocks.add_module(
                f"lstm_{i}",
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0
                )
            )
            self.lstm_blocks.add_module(f"drop_h_{i}", DropHiddenState())
            input_size = factor * hidden_size

        self.dec = nn.Linear(in_features=input_size, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  x (B, T, input_dim)
        Output: logits (B, T, num_classes)
        """
        B, T, _ = x.shape
        x = self.conv_blocks(x.permute(0, 2, 1))  # → (B, C, T')
        x = x.permute(0, 2, 1)                   # → (B, T', C)

        x = self.lstm_blocks(x)                  # → (B, T', H)
        x = self.dec(x)                          # → (B, T', num_classes)

        # Interpolate to match original input length T
        x = F.interpolate(x.permute(0, 2, 1), size=T, mode="linear", align_corners=True)
        return x.permute(0, 2, 1)                # → (B, T, num_classes)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f'\nTrainable parameters: {params}'
