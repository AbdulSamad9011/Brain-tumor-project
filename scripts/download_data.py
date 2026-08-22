"""Pointers + folder scaffolding for the two public datasets this project
is built around. Actual downloads require accepting dataset licenses on
Kaggle / the BraTS challenge site, so this script sets up the expected
directory layout rather than fetching data itself.
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


def scaffold_dirs():
    for cls in ("glioma", "meningioma", "pituitary", "no_tumor"):
        Path(f"data/raw/classification/{cls}").mkdir(parents=True, exist_ok=True)
    Path("data/raw/segmentation").mkdir(parents=True, exist_ok=True)
    print("Created expected directory layout under data/raw/.")
    print(f"\nClassification data: {CLASSIFICATION_DATASET}")
    print(f"Segmentation data:    {SEGMENTATION_DATASET}")
    print(EXPECTED_LAYOUT)


if __name__ == "__main__":
    scaffold_dirs()
