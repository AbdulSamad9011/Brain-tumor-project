"""3D U-Net for tumor segmentation, built on MONAI's implementation so the
architecture is battle-tested rather than hand-rolled.

Swap `UNet` for MONAI's `SwinUNETR` or a full nnU-Net wrapper later —
`forward()` keeps the same (B, 1, D, H, W) -> (B, 1, D, H, W) contract.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class TumorSegmenter(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        logits = self(x)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()
