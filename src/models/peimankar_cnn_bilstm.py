import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

class DropHiddenState(nn.Module):
    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        return x[0]


class PeimankarCnnBilstm(nn.Module):
    """
    https://www.sciencedirect.com/science/article/abs/pii/S0957417420307065
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
        

        self.conv_blocks = nn.Sequential()
        in_channels = input_dim

        dilation = 1
        stride = 1
        padding = [dilation * (kernel-stride) / 2 for kernel in kernel_sizes]

        for i, (out_channels, k) in enumerate(zip(cnn_channels, kernel_sizes)):
            self.conv_blocks.add_module(f"conv_{i}", nn.Conv1d(in_channels, 
                                                               out_channels, 
                                                               kernel_size=k, 
                                                               stride=stride, 
                                                               dilation=dilation))
            self.conv_blocks.add_module(f"relu_{i}", nn.ReLU())
            self.conv_blocks.add_module(f"zp_{i}", nn.ZeroPad1d(int(padding[i])))
            in_channels = out_channels

        self.lstm_blocks = nn.Sequential()
        input_size = cnn_channels[-1]
        factor = 2 if bidirectional else 1

        for i, (hidden_size, layers, dropout) in enumerate(zip(lstm_hidden_sizes, lstm_layers, dropouts)):
            self.lstm_blocks.add_module(f"lstm {i}", nn.LSTM(input_size, 
                                                             hidden_size, 
                                                             num_layers=layers, 
                                                             bidirectional=bidirectional, 
                                                             batch_first=True,
                                                             dropout=dropout))
            self.lstm_blocks.add_module(f"drop_h {i}", DropHiddenState())
            input_size = factor * hidden_size
        

        self.dec = nn.Linear(in_features=factor*lstm_hidden_sizes[-1], out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
            input: (B, T, input_dim)
            output: (B, T, num_classes)
        """
        x = self.conv_blocks(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.lstm_blocks(x)
        x = self.dec(x)
        return x
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

# model = PeimankarCnnBilstm()
# x = torch.rand(32, 100, 1)
# model(x).shape