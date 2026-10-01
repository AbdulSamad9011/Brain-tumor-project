"""Merge classification + segmentation outputs into one structured DiagnosticFinding.

The report generator only ever sees this fused structure — never raw
model internals, tensors, or pixel data.
"""
from __future__ import annotations

from brain_tumor_dx.report.schema import DiagnosticFinding


def fuse(classification: dict, segmentation: dict) -> DiagnosticFinding:
    tumor_present = classification["label"] != "no_tumor" and segmentation["volume_mm3"] > 0

    return DiagnosticFinding(
        tumor_present=tumor_present,
        tumor_type=classification["label"],
        classification_confidence=classification["confidence"],
        class_probabilities=classification["probabilities"],
        tumor_volume_mm3=segmentation["volume_mm3"],
        tumor_centroid=segmentation["centroid"],
    )
