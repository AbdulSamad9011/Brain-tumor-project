"""Preprocessing shared by classification and segmentation paths:
skull-strip -> normalize -> resample -> tensor-ready.

skull_strip_naive uses a percentile-threshold intensity mask suitable for
raw MRI volumes; BraTS-format data (already skull-stripped) skips it and
uses outlier clipping + z-score normalization directly.
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
    """Intensity-threshold skull strip: zeroes out voxels below the given percentile.

    Applied to raw (non-BraTS) volumes before normalization.
    BraTS data is already pre-processed and bypasses this step.
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
    return np.stack([image, image, image], axis=0)  # replicate to 3 channels for ImageNet-pretrained backbone


def preprocess_for_segmentation(volume: np.ndarray, input_size: int) -> np.ndarray:
    """3D volume -> clipped, normalized, resampled, channel-first array.

    Skips skull_strip_naive because BraTS data is already pre-processed.
    The naive percentile threshold destroys 85% of voxels in this dataset.
    Instead, clip to [0, 99.5th percentile] then normalize.
    """
    volume = volume.copy()
    # Clip extreme outliers (keep 0.5-99.5 percentile range)
    p_low, p_high = np.percentile(volume[volume > 0], [0.5, 99.5]) if (volume > 0).any() else (0, 1)
    volume = np.clip(volume, p_low, p_high)
    volume = normalize_intensity(volume)
    volume = resize_volume(volume, input_size)
    return volume[np.newaxis, ...]  # add channel dim
