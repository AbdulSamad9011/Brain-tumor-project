"""Checkpoint loading, kept separate so inference code never touches raw
torch.load calls directly — swap in ONNX/TorchScript export loading here
later without changing any caller."""
from __future__ import annotations

from pathlib import Path

import torch

from brain_tumor_dx.config import settings
from brain_tumor_dx.models.classifier import TumorClassifier
from brain_tumor_dx.models.segmentation import TumorSegmenter

_classifier_cache: TumorClassifier | None = None
_segmenter_cache: TumorSegmenter | None = None


def load_classifier() -> TumorClassifier:
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache

    model = TumorClassifier(num_classes=len(settings.tumor_classes))
    ckpt_path = Path(settings.classifier_ckpt_path)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=settings.device)
        if any(not k.startswith("net.") for k in state.keys()):
            state = {f"net.{k}": v for k, v in state.items()}
        model.load_state_dict(state)
    else:
        print(f"[registry] WARNING: checkpoint not found at {ckpt_path} — check CLASSIFIER_CKPT_PATH in .env. Running with ImageNet weights only.")

    model.to(settings.device).eval()
    _classifier_cache = model
    return model


def load_segmenter(in_channels: int = 1) -> TumorSegmenter:
    global _segmenter_cache
    if _segmenter_cache is not None:
        return _segmenter_cache

    model = TumorSegmenter(in_channels=in_channels)
    ckpt_path = Path(settings.segmentation_ckpt_path)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=settings.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Adapt first conv layer if channel count doesn't match
        first_conv_key = "enc1.conv.0.weight"
        if first_conv_key in state:
            ckpt_in_channels = state[first_conv_key].shape[1]
            if ckpt_in_channels != in_channels:
                if in_channels == 1:
                    # Average across multi-modal channels into single channel
                    state[first_conv_key] = state[first_conv_key].mean(dim=1, keepdim=True)
                else:
                    state[first_conv_key] = state[first_conv_key].repeat(1, in_channels, 1, 1, 1)[:, :in_channels]

        model.load_state_dict(state, strict=False)
    else:
        print(f"[registry] WARNING: checkpoint not found at {ckpt_path} — check SEGMENTATION_CKPT_PATH in .env. Running with random weights only.")

    model.to(settings.device).eval()
    _segmenter_cache = model
    return model
