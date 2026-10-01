"""3D U-Net for volumetric tumor segmentation.

Encoder-decoder architecture with skip connections. Three encoder blocks
downsample via MaxPool3d; three decoder blocks upsample via ConvTranspose3d
and concatenate the matching encoder feature maps. Gradient checkpointing
is supported for memory-efficient training on large volumes.

Swap for MONAI's SwinUNETR or nnU-Net wrapper if needed —
`forward()` keeps the same (B, C, D, H, W) -> (B, 1, D, H, W) contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TumorSegmenter(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv3d(32, out_channels, 1)

    def _pad_to_match(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dd = target.shape[2] - x.shape[2]
        dh = target.shape[3] - x.shape[3]
        dw = target.shape[4] - x.shape[4]
        return F.pad(x, (0, dw, 0, dh, 0, dd))

    def _run_block(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and x.requires_grad:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._run_block(self.enc1, x)
        e2 = self._run_block(self.enc2, self.pool(e1))
        e3 = self._run_block(self.enc3, self.pool(e2))
        b = self._run_block(self.bottleneck, self.pool(e3))

        up3_out = self._pad_to_match(self.up3(b), e3)
        d3 = self._run_block(self.dec3, torch.cat([up3_out, e3], dim=1))

        up2_out = self._pad_to_match(self.up2(d3), e2)
        d2 = self._run_block(self.dec2, torch.cat([up2_out, e2], dim=1))

        up1_out = self._pad_to_match(self.up1(d2), e1)
        d1 = self._run_block(self.dec1, torch.cat([up1_out, e1], dim=1))

        return self.out_conv(d1)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        logits = self(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()
