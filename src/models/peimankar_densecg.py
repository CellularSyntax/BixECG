import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

class Peimankar(nn.Module):
    """
    https://www.sciencedirect.com/science/article/abs/pii/S0957417420307065
    """
    def __init__(self, in_features=1, out_features=4,
                 dropout=0.3, bidirectional=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factor = 2 if bidirectional else 1

        self.enc = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=32, kernel_size=3,),
            nn.ReLU(),
            nn.ZeroPad1d(1),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,),
            nn.ReLU(),
            nn.ZeroPad1d(1),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,),
            nn.ReLU(),
            nn.ZeroPad1d(1),
        )
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=250, bidirectional=bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=factor*250, hidden_size=125, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dec = nn.Linear(in_features=factor*125, out_features=out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
            input: (B, T, in_features)
            output: (B, T, out_features)
        """
        x = self.enc(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.dec(x)
        return x
    

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
