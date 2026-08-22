"""Quick CLI to run the full pipeline on a single local NIfTI file, without
spinning up the API or the Streamlit app — useful for smoke-testing a new
checkpoint.

Run from repo root: python scripts/run_pipeline_cli.py path/to/scan.nii.gz
"""
from __future__ import annotations

import argparse
import asyncio

from brain_tumor_dx.data.io import load_nifti
from brain_tumor_dx.pipeline import run_pipeline


async def main(nifti_path: str):
    volume = load_nifti(nifti_path)
    mid_slice = volume[volume.shape[0] // 2]

    report = await run_pipeline(slice_2d=mid_slice, volume_3d=volume)

    print(f"\n=== {report.summary} ===\n")
    print(report.narrative)
    print(f"\n[confidence note] {report.confidence_note}")
    print(f"[recommendation]  {report.recommendation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nifti_path", help="Path to a .nii/.nii.gz volume")
    args = parser.parse_args()
    asyncio.run(main(args.nifti_path))
