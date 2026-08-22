"""Grad-CAM overlay for the classifier — lets a clinician see *why* the
model predicted a given class instead of trusting a bare label."""
from __future__ import annotations

import numpy as np
import torch

from brain_tumor_dx.config import settings


def gradcam_overlay(model, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """Returns a (H, W) heatmap in [0, 1], same spatial size as the input.

    Uses the `grad-cam` package's implementation rather than a hand-rolled
    hook. The target layer below assumes a ResNet backbone — change it if
    you swap architectures in models/classifier.py.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    target_layer = model.net.layer4[-1]  # last conv block of ResNet50
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor.to(settings.device), targets=targets)
    return grayscale_cam[0]
