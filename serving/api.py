"""FastAPI service exposing the pipeline over HTTP.

POST /predict expects a multipart upload of a NIfTI volume; a representative
mid-axial slice is extracted server-side for the classifier while the full
volume feeds the segmenter. Swap in DICOM upload support via
data/io.py::load_dicom_series if your source scans come from a PACS instead.

Run from the repo root: uvicorn serving.api:app --reload
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from brain_tumor_dx.data.io import load_nifti
from brain_tumor_dx.pipeline import run_pipeline
from serving.schemas import PredictResponse

app = FastAPI(title="Brain Tumor Diagnosis API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "Expected a .nii or .nii.gz file")

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        volume = load_nifti(tmp_path)
        mid_slice = volume[volume.shape[0] // 2]  # representative axial slice for the classifier

        report = await run_pipeline(slice_2d=mid_slice, volume_3d=volume)
        return PredictResponse(report=report)
    finally:
        tmp_path.unlink(missing_ok=True)
