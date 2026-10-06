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

class BiXLSTMClassifierFusion(nn.Module):
    def __init__(
        self,
        seq_length: int = 1000,
        input_dim: int = 1,
        embedding_dim: int = 64,
        num_blocks: int = 4,
        num_heads: int = 2,
        conv1d_kernel_size: int = 3,
        proj_factor: float = 1.1,
        slstm_at: list = [1],
        use_slstm: bool = True,
        dropout: float = 0.3,
        num_classes: int = 5,
        feature_dim: int = 40,     # Number of feature columns
        slstm_backend: str = None,
    ):
        super().__init__()

        if slstm_backend is None:
            slstm_backend = "cuda" if torch.cuda.is_available() else "vanilla"

        self.input_proj = nn.Linear(input_dim, embedding_dim)

        # Shared xLSTM block
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
            feedforward=FeedForwardConfig(proj_factor=proj_factor, act_fn="gelu"),
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

        # Global pooling for signal representation
        self.signal_pooling = nn.AdaptiveAvgPool1d(1)

        # Feature branch
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )

        # Final fusion classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * embedding_dim + embedding_dim),  # 3 * embedding_dim
            nn.Dropout(dropout),
            nn.Linear(3 * embedding_dim, num_classes)
        )

    def forward(self, ecg_seq, features):
        # ecg_seq: (B, T, 1), features: (B, F)
        x = self.input_proj(ecg_seq)           # (B, T, D)
        x_forward = self.xlstm_stack(x)
        x_backward = self.xlstm_stack(torch.flip(x, dims=[1]))
        x_bidir = torch.cat([x_forward, torch.flip(x_backward, dims=[1])], dim=-1)  # (B, T, 2D)

        # Global pooling over time
        x_signal = x_bidir.permute(0, 2, 1)    # (B, 2D, T)
        x_signal = self.signal_pooling(x_signal).squeeze(-1)  # (B, 2D)

        # Feature branch
        x_feat = self.feature_mlp(features)    # (B, D)

        # Fuse
        x_combined = torch.cat([x_signal, x_feat], dim=-1)  # (B, 2D + D)
        out = self.classifier(x_combined)                   # (B, num_classes)
        return out
