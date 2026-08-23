"""Preprocessing shared by classification and segmentation paths:
skull-strip -> normalize -> resample -> tensor-ready.

Skull stripping and bias-field correction are the two steps most worth
swapping for a real implementation (HD-BET / ANTsPyNet) before this
touches real patient data — the stub below falls back to a naive
intensity-threshold mask so the pipeline is runnable end-to-end without
extra binaries or GPU-only tooling.
"""
from __future__ import annotations

import numpy as np


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize non-zero (foreground) voxels."""
    mask = volume > 0
    if not mask.any():
        return volume
    mean, std = volume[mask].mean(), volume[mask].std() + 1e-8
    out = volume.copy()
    out[mask] = (volume[mask] - mean) / std
    return out


def skull_strip_naive(volume: np.ndarray, threshold_percentile: float = 5.0) -> np.ndarray:
    """Placeholder skull-strip: zeroes out low-intensity background voxels.

    TODO: replace with HD-BET or ANTsPyNet for anything beyond a demo —
    a percentile threshold is not a real brain mask.
    """
    threshold = np.percentile(volume, threshold_percentile)
    stripped = volume.copy()
    stripped[volume < threshold] = 0
    return stripped


def resize_2d(image: np.ndarray, size: int) -> np.ndarray:
    from skimage.transform import resize

    return resize(image, (size, size), anti_aliasing=True, preserve_range=True).astype(np.float32)


def resize_volume(volume: np.ndarray, size: int, order: int = 1, anti_aliasing: bool = True) -> np.ndarray:
    from skimage.transform import resize

    return resize(
        volume, (size, size, size), order=order, anti_aliasing=anti_aliasing, preserve_range=True
    ).astype(np.float32)


def preprocess_for_classifier(image: np.ndarray, input_size: int) -> np.ndarray:
    """2D slice -> normalized, resized, channel-first array ready for the classifier."""
    image = normalize_intensity(image)
    image = resize_2d(image, input_size)
    return np.stack([image, image, image], axis=0)  # fake 3-channel for imagenet-pretrained backbones


def preprocess_for_segmentation(volume: np.ndarray, input_size: int) -> np.ndarray:
    """3D volume -> skull-stripped, normalized, resampled, channel-first array."""
    volume = skull_strip_naive(volume)
    volume = normalize_intensity(volume)
    volume = resize_volume(volume, input_size)
    return volume[np.newaxis, ...]  # add channel dim
