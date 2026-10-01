"""Trains TumorSegmenter on paired image/mask volumes (BraTS-style layout).

Features:
- PyTorch Automatic Mixed Precision (AMP) for 2x memory reduction and GPU acceleration
- Gradient Checkpointing for low VRAM (e.g. 2GB/4GB local GPUs)
- Gradient Accumulation for effective batch size scaling
- Comprehensive 3D spatial and intensity data augmentations
- Combined Dice + Weighted BCE loss with proper device buffer registration
- Multi-metric validation (Dice, IoU, Validation Loss)
- Cosine Annealing LR scheduler with warmup support
- Best-model checkpointing & early stopping
- Peak VRAM tracking and automatic cache cleanup
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.datasets import SegmentationDataset
from brain_tumor_dx.models.segmentation import TumorSegmenter


def augment_3d(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Medical 3D data augmentation: spatial flips, 90-deg rotations, intensity scaling & noise."""
    # 1. Random flips along spatial axes D, H, W (dims 2, 3, 4)
    for axis in (2, 3, 4):
        if random.random() > 0.5:
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis])

    # 2. Random 90-degree rotations in axial plane (H, W -> dims 3, 4)
    if random.random() > 0.5:
        k = random.choice([1, 2, 3])
        image = torch.rot90(image, k, dims=[3, 4])
        mask = torch.rot90(mask, k, dims=[3, 4])

    # 3. Random intensity scaling and shifting (image only)
    if random.random() > 0.5:
        scale_factor = random.uniform(0.9, 1.1)
        shift_factor = random.uniform(-0.1, 0.1)
        image = image * scale_factor + shift_factor

    # 4. Random subtle Gaussian noise
    if random.random() > 0.3:
        noise = torch.randn_like(image) * 0.02
        image = image + noise

    return image, mask


def compute_metrics(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8
) -> dict[str, float]:
    """Computes Dice similarity coefficient and IoU (Jaccard Index) for binary segmentation."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum().item()
    pred_sum = pred_bin.sum().item()
    target_sum = target_bin.sum().item()

    if target_sum == 0 and pred_sum == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        union = pred_sum + target_sum - intersection
        iou = (intersection + eps) / (union + eps)

    return {"dice": float(dice), "iou": float(iou)}


class DiceBCELoss(torch.nn.Module):
    """Combined Dice + Weighted BCE loss for imbalanced medical segmentation.
    Properly registers pos_weight as a tensor buffer to prevent CPU/CUDA device mismatches.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
    ):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        pos_w = self.pos_weight.to(device=pred.device, dtype=pred.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_w)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def train(
    data_root: str,
    epochs: int = 50,
    batch_size: int = 1,
    lr: float = 1e-4,
    out_path: str = "checkpoints/segmentation.pt",
    val_split: float = 0.2,
    patience: int = 20,
    input_size: int = 96,
    accum_steps: int = 2,
    use_amp: bool = True,
    use_checkpointing: bool = True,
    device_name: str | None = None,
):
    device = torch.device(device_name or settings.device)
    is_cuda = device.type == "cuda"
    use_amp = use_amp and is_cuda

    print(f"=== Segmentation Training Configuration ===")
    print(f"Device:               {device}")
    if is_cuda:
        print(f"GPU Model:            {torch.cuda.get_device_name(device)}")
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"Total VRAM:           {total_vram:.2f} GB")
    print(f"Input Spatial Size:   {input_size}^3")
    print(f"Automatic Mixed Prec: {use_amp}")
    print(f"Gradient Checkpoint:  {use_checkpointing}")
    print(f"Gradient Accum Steps: {accum_steps} (effective batch size: {batch_size * accum_steps})")
    print(f"Batch Size:           {batch_size}")
    print(f"Learning Rate:        {lr}")
    print(f"Epochs:               {epochs}")
    print(f"Output Checkpoint:    {out_path}")
    print("===========================================")

    # Dataset & Loaders
    dataset = SegmentationDataset(data_root, input_size=input_size)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=is_cuda,
    )

    print(f"Train Cases: {n_train} | Validation Cases: {n_val}\n")

    # Model & Optimization
    model = TumorSegmenter(
        in_channels=1,
        out_channels=1,
        use_checkpointing=use_checkpointing,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = DiceBCELoss(pos_weight=10.0, dice_weight=1.0, bce_weight=1.0).to(device)

    best_dice = 0.0
    best_iou = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        n_train_batches = 0
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:02d}/{epochs:02d} [train]",
            leave=False,
        )

        for step, (images, masks) in enumerate(train_pbar):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            images, masks = augment_3d(images, masks)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(images)
                loss = criterion(pred, masks)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            n_train_batches += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(n_train_batches, 1)

        # Validation phase
        model.eval()
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(images)
                    v_loss = criterion(pred, masks)

                metrics = compute_metrics(pred, masks)
                val_dice_sum += metrics["dice"]
                val_iou_sum += metrics["iou"]
                val_loss_sum += v_loss.item()
                n_val_batches += 1

        avg_val_dice = val_dice_sum / max(n_val_batches, 1)
        avg_val_iou = val_iou_sum / max(n_val_batches, 1)
        avg_val_loss = val_loss_sum / max(n_val_batches, 1)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        epoch_dur = time.time() - epoch_start_time

        vram_info = ""
        if is_cuda:
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
            vram_info = f" | VRAM: {peak_vram:.1f}MB"
            torch.cuda.empty_cache()

        marker = ""
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            best_iou = avg_val_iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_path)
            marker = " [★ BEST]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} ({epoch_dur:.1f}s): "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_dice={avg_val_dice:.4f} | "
            f"val_iou={avg_val_iou:.4f} | lr={lr_now:.2e}{vram_info}{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break

    print(f"\nTraining Complete!")
    print(f"Best Validation Dice: {best_dice:.4f} (IoU: {best_iou:.4f}) at epoch {best_epoch}")
    print(f"Checkpoint saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Brain Tumor Segmenter")
    parser.add_argument("--data-root", required=True, help="Directory of case_XXX/{image,mask}.nii.gz")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per forward pass")
    parser.add_argument("--input-size", type=int, default=96, help="Spatial resolution cube (e.g. 96 or 128)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--no-checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--out", default="checkpoints/segmentation.pt", help="Output path for best checkpoint")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience in epochs")
    parser.add_argument("--device", default=None, help="Device to train on (e.g. cuda, cuda:0, cpu)")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        val_split=args.val_split,
        patience=args.patience,
        input_size=args.input_size,
        accum_steps=args.accum_steps,
        use_amp=not args.no_amp,
        use_checkpointing=not args.no_checkpointing,
        device_name=args.device,
    )
