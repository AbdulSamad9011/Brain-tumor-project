"""End-to-end orchestration: preprocessing -> classify + segment (concurrently)
-> fuse -> report. The scaffold's equivalent of graph.py in the research-agent
project, minus LangGraph — there's no branching plan to route here, so a
plain async function is enough. Swap in LangGraph later if you add more
stages (e.g. a critic node, or a human-in-the-loop re-scan request loop).
"""
from __future__ import annotations

import asyncio

import numpy as np

from brain_tumor_dx.inference.classify import classify_slice
from brain_tumor_dx.inference.fusion import fuse
from brain_tumor_dx.inference.segment import segment_volume
from brain_tumor_dx.report.generator import generate_report
from brain_tumor_dx.report.schema import DiagnosticReport


async def run_pipeline(slice_2d: np.ndarray, volume_3d: np.ndarray) -> DiagnosticReport:
    """slice_2d: a representative 2D slice for classification.
    volume_3d: the full 3D volume for segmentation.

    Accepting both lets the classifier run on the dataset's native 2D
    format while segmentation still gets full volumetric context — merge
    these into one 3D-only path once the classifier is retrained on
    volumes instead of the Kaggle 2D slice dataset.
    """
    classification_task = asyncio.to_thread(classify_slice, slice_2d)
    segmentation_task = asyncio.to_thread(segment_volume, volume_3d)

    classification, segmentation = await asyncio.gather(classification_task, segmentation_task)

    finding = fuse(classification, segmentation)
    report = await asyncio.to_thread(generate_report, finding)
    return report
