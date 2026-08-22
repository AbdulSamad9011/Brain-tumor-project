"""Standard metrics for both tasks — kept dependency-light (numpy only) so
they can run in a CI test without a GPU."""
from __future__ import annotations

import numpy as np


def dice_coefficient(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + true.sum() + eps))


def iou(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float((intersection + eps) / (union + eps))


def classification_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        matrix[l, p] += 1
    return matrix
