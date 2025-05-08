import torch
import torch.nn as nn
import os
import sys

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class xLSTM(nn.Module):
    def __init__(
        self,
        seq_length: int = 500,
        input_dim: int = 1,
        embedding_dim: int = 64,
        num_blocks: int = 4,
        num_heads: int = 2,
        conv1d_kernel_size: int = 3,
        proj_factor: float = 1.1,
        slstm_at: list = [1],
        use_slstm: bool = True,
        dropout: float = 0.3,
        num_classes: int = 4,
        slstm_backend: str = None, 
    ):
        super().__init__()
        # Auto-detect backend if not specified
        if slstm_backend is None:
            slstm_backend = "cuda" if torch.cuda.is_available() else "vanilla"

        self.input_proj = nn.Linear(input_dim, embedding_dim)

        mlstm_cfg = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=2,
                num_heads=num_heads,
            )
        )

        slstm_cfg = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=slstm_backend,
                num_heads=num_heads,
                conv1d_kernel_size=conv1d_kernel_size,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(
                proj_factor=proj_factor,
                act_fn="gelu"
            ),
        ) if use_slstm else None

        xlstm_cfg = xLSTMBlockStackConfig(
            context_length=seq_length,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            mlstm_block=mlstm_cfg,
            slstm_block=slstm_cfg,
            slstm_at=slstm_at if use_slstm else [],
        )

        self.xlstm_stack = xLSTMBlockStack(xlstm_cfg)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x shape: (B, T, 1)
        x = self.input_proj(x)                     # (B, T, D)
        x = self.xlstm_stack(x)                    # (B, T, D)
        x = self.norm(self.dropout(x))             # (B, T, D)
        logits = self.output_proj(x)               # (B, T, C)
        return logits
    
class BiXLSTM(nn.Module): # with shared weights
    def __init__(
        self,
        seq_length: int = 500,
        input_dim: int = 1,
        embedding_dim: int = 64,
        num_blocks: int = 4,
        num_heads: int = 2,
        conv1d_kernel_size: int = 3,
        proj_factor: float = 1.1,
        slstm_at: list = [1],
        use_slstm: bool = True,
        dropout: float = 0.3,
        num_classes: int = 4,
        slstm_backend: str = None, 
    ):
        super().__init__()
        # Auto-detect backend if not specified
        if slstm_backend is None:
            slstm_backend = "cuda" if torch.cuda.is_available() else "vanilla"

        self.input_proj = nn.Linear(input_dim, embedding_dim)

        # Only ONE shared xLSTM block stack
        mlstm_cfg = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=2,
                num_heads=num_heads,
            )
        )

        slstm_cfg = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=slstm_backend,
                num_heads=num_heads,
                conv1d_kernel_size=conv1d_kernel_size,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(
                proj_factor=proj_factor,
                act_fn="gelu"
            ),
        ) if use_slstm else None

        xlstm_cfg = xLSTMBlockStackConfig(
            context_length=seq_length,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            mlstm_block=mlstm_cfg,
            slstm_block=slstm_cfg,
            slstm_at=slstm_at if use_slstm else [],
        )

        self.xlstm_stack = xLSTMBlockStack(xlstm_cfg)  # Only ONE shared stack

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(2 * embedding_dim)
        self.output_proj = nn.Linear(2 * embedding_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)  # (B, T, D)

        x_forward = self.xlstm_stack(x.clone())  # (B, T, D)

        x_backward = self.xlstm_stack(torch.flip(x, dims=[1]))  # reverse along time
        x_backward = torch.flip(x_backward, dims=[1])           # unflip back

        x = torch.cat([x_forward, x_backward], dim=-1)  # (B, T, 2D)

        x = self.norm(self.dropout(x))
        logits = self.output_proj(x)
        return logits
