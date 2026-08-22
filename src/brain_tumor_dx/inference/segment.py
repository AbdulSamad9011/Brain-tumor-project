"""Run the segmentation model on a preprocessed volume and derive simple
clinical-adjacent measurements from the predicted mask."""
from __future__ import annotations

import numpy as np
import torch

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.preprocessing import preprocess_for_segmentation
from brain_tumor_dx.models.registry import load_segmenter


def segment_volume(volume: np.ndarray, voxel_volume_mm3: float = 1.0) -> dict:
    """Returns {"mask": np.ndarray, "volume_mm3": float, "centroid": (z, y, x) | None}."""
    model = load_segmenter()
    array = preprocess_for_segmentation(volume, settings.segmentation_input_size)
    tensor = torch.from_numpy(array).unsqueeze(0).to(settings.device)  # add batch dim

    mask = model.predict_mask(tensor).squeeze(0).squeeze(0).cpu().numpy()  # (D, H, W)
    voxel_count = int(mask.sum())

    if voxel_count == 0:
        return {"mask": mask, "volume_mm3": 0.0, "centroid": None}

    coords = np.argwhere(mask > 0)
    centroid = tuple(coords.mean(axis=0).round(1))

    return {
        "mask": mask,
        "volume_mm3": float(voxel_count * voxel_volume_mm3),
        "centroid": centroid,
    }
