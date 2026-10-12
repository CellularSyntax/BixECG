"""BixECGNet: the 2,522-parameter Bi-mLSTM delineator reported in the paper.

This is the exact architecture behind both published weight sets
(``bixecg_mlstm_2522_detaug.pt`` and ``bixecg_mlstm_2522.pt``):

* ``input_proj``: Linear(1 -> 8)
* one shared ``xLSTMBlockStack`` of 3 mLSTM blocks (embedding dim 8, 1 head,
  causal depthwise conv kernel 11, up-projection rounded to 16, post-blocks norm)
* the same stack is run on the sequence and on its time reversal (weight sharing)
* ``output_proj``: Linear(16 -> 4) over concat(forward, backward)

Input: ``(batch, 300, 1)`` single-lead ECG at 250 Hz, band-passed 0.5-40 Hz and
per-window z-normalised. Output: ``(batch, 300, 4)`` logits over
{No-Wave, P-wave, QRS, T-wave}.
"""

import torch
import torch.nn as nn
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig

__all__ = ["BixECGNet", "BIXECG_2522_KWARGS"]

#: Constructor arguments of the selected 2,522-parameter configuration.
BIXECG_2522_KWARGS = dict(
    seq_length=300,
    input_dim=1,
    embedding_dim=8,
    num_blocks=3,
    num_heads=1,
    conv1d_kernel_size=11,
    num_classes=4,
)


class BixECGNet(nn.Module):
    """Weight-shared bidirectional mLSTM for per-sample ECG delineation."""

    def __init__(
        self,
        seq_length: int = 300,
        input_dim: int = 1,
        embedding_dim: int = 8,
        num_blocks: int = 3,
        num_heads: int = 1,
        conv1d_kernel_size: int = 11,
        num_classes: int = 4,
        proj_factor: float = 1.0,
        round_up: bool = True,
        round_mult: int = 16,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        mcfg = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=proj_factor,
                round_proj_up_dim_up=round_up,
                round_proj_up_to_multiple_of=round_mult,
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=2,
                num_heads=num_heads,
            )
        )
        cfg = xLSTMBlockStackConfig(
            context_length=seq_length,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            mlstm_block=mcfg,
            slstm_block=None,
            slstm_at=[],
        )
        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(2 * embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        xf = self.xlstm_stack(x)
        xb = torch.flip(self.xlstm_stack(torch.flip(x, dims=[1])), dims=[1])
        return self.output_proj(self.dropout(torch.cat([xf, xb], dim=-1)))
