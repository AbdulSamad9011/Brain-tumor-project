"""Fine-tunes TumorClassifier on the Kaggle-style directory dataset.

Includes proper train/val split, label smoothing, LR scheduling,
AMP support, and best-model checkpointing to prevent data leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.datasets import ClassificationDataset
from brain_tumor_dx.models.classifier import TumorClassifier


def train(
    data_root: str,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    out_path: str = "checkpoints/classifier.pt",
    val_split: float = 0.15,
    patience: int = 10,
    use_amp: bool = True,
    device_name: str | None = None,
):
    device = torch.device(device_name or settings.device)
    is_cuda = device.type == "cuda"
    use_amp = use_amp and is_cuda

    print(f"=== Classifier Training Configuration ===")
    print(f"Device:               {device}")
    if is_cuda:
        print(f"GPU Model:            {torch.cuda.get_device_name(device)}")
    print(f"Automatic Mixed Prec: {use_amp}")
    print(f"Batch Size:           {batch_size}")
    print(f"Learning Rate:        {lr}")
    print(f"Epochs:               {epochs}")
    print(f"Output Checkpoint:    {out_path}")
    print("=========================================")

    dataset = ClassificationDataset(data_root)

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if not is_cuda else 0,
        pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if not is_cuda else 0,
        pin_memory=is_cuda,
    )

    print(f"Train: {n_train} | Val: {n_val}\n")

    model = TumorClassifier(num_classes=len(settings.tumor_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{epochs:02d} [train]", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(images)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(images)
                    loss = criterion(out, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_path)
            marker = " [★ BEST]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d}: "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={lr_now:.2e}{marker}"
        )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tumor Classifier")
    parser.add_argument("--data-root", required=True, help="Directory with one subfolder per class")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/classifier.pt")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda, cuda:0, cpu)")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        val_split=args.val_split,
        patience=args.patience,
        use_amp=not args.no_amp,
        device_name=args.device,
    )
