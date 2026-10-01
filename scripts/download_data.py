"""Sets up the expected data directory layout for the two training datasets.

Actual downloads require accepting dataset licenses on Kaggle / the BraTS
challenge site, so this script creates the expected directory tree and
prints download instructions. Run once before your first training run.
"""
from __future__ import annotations

from pathlib import Path

CLASSIFICATION_DATASET = (
    "Kaggle 'Brain Tumor MRI Dataset' — "
    "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
)
SEGMENTATION_DATASET = (
    "BraTS (Brain Tumor Segmentation Challenge) — "
    "https://www.med.upenn.edu/cbica/brats/"
)

EXPECTED_LAYOUT = """
data/raw/classification/
    glioma/*.jpg
    meningioma/*.jpg
    pituitary/*.jpg
    no_tumor/*.jpg

data/raw/segmentation/
    case_001/image.nii.gz
    case_001/mask.nii.gz
    case_002/...
"""


def setup_dirs():
    for cls in ("glioma", "meningioma", "pituitary", "no_tumor"):
        Path(f"data/raw/classification/{cls}").mkdir(parents=True, exist_ok=True)
    Path("data/raw/segmentation").mkdir(parents=True, exist_ok=True)
    print("Created expected directory layout under data/raw/.")
    print(f"\nClassification data: {CLASSIFICATION_DATASET}")
    print(f"Segmentation data:    {SEGMENTATION_DATASET}")
    print(EXPECTED_LAYOUT)


if __name__ == "__main__":
    setup_dirs()
