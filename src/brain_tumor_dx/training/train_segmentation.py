"""Trains TumorSegmenter on paired image/mask volumes (BraTS-style layout).

Includes:
- Dice + BCE combined loss for class imbalance
- Train/val split with seed for reproducibility
- Data augmentation (random flips)
- Cosine annealing LR scheduler
- Best-model checkpointing on validation Dice
- Early stopping
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.datasets import SegmentationDataset
from brain_tumor_dx.models.segmentation import TumorSegmenter


def augment_3d(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Random augmentation for 3D volumes: flips along random axes."""
    # Random flip along each spatial axis (D, H, W)
    for axis in [2, 3, 4]:  # dims 2,3,4 are D,H,W in (B,C,D,H,W)
        if random.random() > 0.5:
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis])
    return image, mask


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum().item()
    return (2 * intersection + 1e-8) / (pred_bin.sum().item() + target.sum().item() + 1e-8)


class DiceBCELoss(torch.nn.Module):
    """Combined Dice + weighted BCE loss for imbalanced segmentation."""
    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice(pred, target) + self.bce(pred, target)


def train(data_root: str, epochs: int, batch_size: int, lr: float, out_path: str,
          val_split: float = 0.2, patience: int = 20):
    dataset = SegmentationDataset(data_root)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Train: {n_train} | Val: {n_val}")

    model = TumorSegmenter().to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DiceBCELoss(pos_weight=10.0)

    best_dice = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs} [train]", leave=False):
            images, masks = images.to(settings.device), masks.to(settings.device)
            images, masks = augment_3d(images, masks)

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        avg_loss = running_loss / max(n_batches, 1)

        model.eval()
        val_dice_sum = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(settings.device), masks.to(settings.device)
                pred = model(images)
                val_dice_sum += compute_dice(pred, masks)
        avg_dice = val_dice_sum / len(val_loader)

        scheduler.step()

        marker = ""
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_path)
            marker = " [BEST]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch + 1}: loss={avg_loss:.4f} | val_dice={avg_dice:.4f} | lr={lr_now:.2e}{marker}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\nBest val Dice: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory of case_XXX/{image,mask}.nii.gz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/segmentation.pt")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    train(args.data_root, args.epochs, args.batch_size, args.lr, args.out,
          args.val_split, args.patience)
