"""PyTorch Dataset wrappers for the two training tasks.

ClassificationDataset expects a directory-per-class layout, matching the
public Kaggle "Brain Tumor MRI Dataset":
    root/
      glioma/*.jpg
      meningioma/*.jpg
      pituitary/*.jpg
      no_tumor/*.jpg

SegmentationDataset expects paired image/mask NIfTI volumes, matching a
BraTS-style layout:
    root/
      case_001/image.nii.gz
      case_001/mask.nii.gz
      case_002/...
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.io import load_image_2d, load_nifti
from brain_tumor_dx.data.preprocessing import preprocess_for_classifier, preprocess_for_segmentation


class ClassificationDataset(Dataset):
    def __init__(self, root: str | Path, input_size: int | None = None):
        self.root = Path(root)
        self.input_size = input_size or settings.classifier_input_size
        self.classes = list(settings.tumor_classes)
        self.samples: list[tuple[Path, int]] = []
        for label_idx, cls in enumerate(self.classes):
            for f in (self.root / cls).glob("*"):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((f, label_idx))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root}. Expected one subfolder per class: {self.classes}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = load_image_2d(path)
        array = preprocess_for_classifier(image, self.input_size)
        return torch.from_numpy(array), label


class SegmentationDataset(Dataset):
    def __init__(self, root: str | Path, input_size: int | None = None):
        self.root = Path(root)
        self.input_size = input_size or settings.segmentation_input_size
        self.cases = sorted(p for p in self.root.iterdir() if p.is_dir())

        if not self.cases:
            raise FileNotFoundError(f"No case folders found under {self.root}")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_dir = self.cases[idx]
        image = load_nifti(case_dir / "image.nii.gz")
        mask = load_nifti(case_dir / "mask.nii.gz")

        image = preprocess_for_segmentation(image, self.input_size)
        mask = preprocess_for_segmentation(mask, self.input_size)
        mask = (mask > 0.5).astype(np.float32)  # binarize after resampling

        return torch.from_numpy(image), torch.from_numpy(mask)
