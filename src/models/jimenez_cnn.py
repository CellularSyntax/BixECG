import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Two-layer 1D convolutional block with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class JimenezCNN1D(nn.Module):
    """
    U-Net–like 1D CNN model for ECG delineation, as described in
    Jimenez-Perez et al., 2024 (Frontiers in Cardiovascular Medicine).
    """
    def __init__(self,
                 input_dim=1,
                 base_channels=32,
                 num_classes=4):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock1D(input_dim, base_channels)
        self.enc2 = ConvBlock1D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock1D(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock1D(base_channels * 4, base_channels * 8)
        self.enc5 = ConvBlock1D(base_channels * 8, base_channels * 8)

        self.pool = nn.MaxPool1d(2)

        # Decoder
        self.up4 = nn.ConvTranspose1d(base_channels * 8, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock1D(base_channels * 8 + base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1D(base_channels * 4 + base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose1d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(base_channels * 2 + base_channels * 2, base_channels)

        self.up1 = nn.ConvTranspose1d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(base_channels + base_channels, base_channels)

        self.out_conv = nn.Conv1d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # Decoder
        d4 = self.up4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)  # (B, num_classes, T)
        return out.permute(0, 2, 1)  # → (B, T, num_classes)
