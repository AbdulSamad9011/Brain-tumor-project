"""Load MRI/CT volumes from disk — DICOM series or NIfTI files — into numpy arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a .nii/.nii.gz volume as a numpy array (Z, Y, X)."""
    import nibabel as nib

    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float32)


def load_dicom_series(directory: str | Path) -> np.ndarray:
    """Load a directory of DICOM slices as a single 3D numpy volume, sorted by
    InstanceNumber so slices come out in anatomical order."""
    import pydicom

    directory = Path(directory)
    files = sorted(directory.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {directory}")

    slices = [pydicom.dcmread(str(f)) for f in files]
    slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0)))
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    return volume


def load_image_2d(path: str | Path) -> np.ndarray:
    """Load a single 2D slice (png/jpg) — used by the classification dataset,
    which the public Kaggle Brain Tumor MRI Dataset ships as flat images
    rather than volumes."""
    from skimage.io import imread

    return np.asarray(imread(str(path), as_gray=True), dtype=np.float32)
