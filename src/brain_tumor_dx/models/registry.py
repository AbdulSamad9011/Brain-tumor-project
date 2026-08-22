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
        model.load_state_dict(state)
    else:
        print(f"[registry] No checkpoint at {ckpt_path} — using ImageNet-initialized weights only.")

    model.to(settings.device).eval()
    _classifier_cache = model
    return model


def load_segmenter() -> TumorSegmenter:
    global _segmenter_cache
    if _segmenter_cache is not None:
        return _segmenter_cache

    model = TumorSegmenter()
    ckpt_path = Path(settings.segmentation_ckpt_path)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=settings.device)
        model.load_state_dict(state)
    else:
        print(f"[registry] No checkpoint at {ckpt_path} — using randomly-initialized weights only.")

    model.to(settings.device).eval()
    _segmenter_cache = model
    return model
