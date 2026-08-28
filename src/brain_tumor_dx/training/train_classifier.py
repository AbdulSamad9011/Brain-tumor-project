"""Fine-tunes TumorClassifier on the Kaggle-style directory dataset.

Includes proper train/val split, label smoothing, LR scheduling,
and best-model checkpointing to prevent data leakage.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.datasets import ClassificationDataset
from brain_tumor_dx.models.classifier import TumorClassifier


def train(data_root: str, epochs: int, batch_size: int, lr: float, out_path: str,
          val_split: float = 0.15, patience: int = 10):
    dataset = ClassificationDataset(data_root)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {n_train} | Val: {n_val}")

    model = TumorClassifier(num_classes=len(settings.tumor_classes)).to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs} [train]", leave=False):
            images, labels = images.to(settings.device), labels.to(settings.device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(settings.device), labels.to(settings.device)
                out = model(images)
                loss = criterion(out, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), out_path)
            marker = " [BEST]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={lr_now:.2e}{marker}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    print(f"\nBest val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory with one subfolder per class")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/classifier.pt")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    train(args.data_root, args.epochs, args.batch_size, args.lr, args.out,
          args.val_split, args.patience)
