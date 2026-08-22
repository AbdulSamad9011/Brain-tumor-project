"""Tumor-type classifier: a fine-tuned torchvision backbone.

ResNet50 by default — swap for EfficientNet or a ViT via `backbone=`
without touching any other file, since everything downstream only cares
about `forward()` returning per-class logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class TumorClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Linear(in_features, num_classes)
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self(x)
        return torch.softmax(logits, dim=-1)
