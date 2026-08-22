"""Lightweight tests that don't require trained checkpoints or real patient
data — verify shapes and metric math, not clinical accuracy."""
from __future__ import annotations

import numpy as np

from brain_tumor_dx.data.preprocessing import preprocess_for_classifier, preprocess_for_segmentation
from brain_tumor_dx.evaluation.metrics import dice_coefficient, iou


def test_dice_perfect_overlap():
    mask = np.ones((10, 10), dtype=bool)
    assert dice_coefficient(mask, mask) == 1.0


def test_dice_no_overlap():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:5] = True
    b[5:] = True
    assert dice_coefficient(a, b) == 0.0


def test_iou_partial_overlap():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:6] = True
    b[4:] = True
    score = iou(a, b)
    assert 0.0 < score < 1.0


def test_preprocess_for_classifier_shape():
    image = np.random.rand(200, 180).astype(np.float32)
    out = preprocess_for_classifier(image, input_size=224)
    assert out.shape == (3, 224, 224)


def test_preprocess_for_segmentation_shape():
    volume = np.random.rand(50, 60, 55).astype(np.float32)
    out = preprocess_for_segmentation(volume, input_size=64)
    assert out.shape == (1, 64, 64, 64)
