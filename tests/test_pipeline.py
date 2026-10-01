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
    assert dice_coefficient(a, b) < 1e-6


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


def test_dice_bce_loss():
    import torch
    from brain_tumor_dx.training.train_segmentation import DiceBCELoss

    criterion = DiceBCELoss(pos_weight=10.0)
    pred = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()

    loss = criterion(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None

    if torch.cuda.is_available():
        criterion_gpu = DiceBCELoss(pos_weight=10.0).cuda()
        pred_gpu = pred.detach().cuda().requires_grad_(True)
        target_gpu = target.cuda()
        loss_gpu = criterion_gpu(pred_gpu, target_gpu)
        assert torch.isfinite(loss_gpu)
        loss_gpu.backward()
        assert pred_gpu.grad is not None


def test_tumor_segmenter_checkpointing():
    import torch
    from brain_tumor_dx.models.segmentation import TumorSegmenter

    model = TumorSegmenter(in_channels=1, out_channels=1, use_checkpointing=True)
    model.train()
    x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (1, 1, 32, 32, 32)
    out.sum().backward()
    assert x.grad is not None


def test_augment_3d():
    import torch
    from brain_tumor_dx.training.train_segmentation import augment_3d

    image = torch.rand(1, 1, 32, 32, 32)
    mask = torch.randint(0, 2, (1, 1, 32, 32, 32)).float()
    aug_img, aug_mask = augment_3d(image, mask)
    assert aug_img.shape == image.shape
    assert aug_mask.shape == mask.shape


def test_load_segmenter():
    from brain_tumor_dx.models.registry import load_segmenter

    segmenter = load_segmenter()
    assert segmenter is not None
