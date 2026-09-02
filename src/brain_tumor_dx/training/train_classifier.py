"""Fine-tunes TumorClassifier on the Kaggle-style directory dataset.

Minimal reference loop — swap in MLflow logging, LR scheduling, or early
stopping before treating this as production training code.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.datasets import ClassificationDataset
from brain_tumor_dx.models.classifier import TumorClassifier


def train(data_root: str, epochs: int, batch_size: int, lr: float, out_path: str):
    dataset = ClassificationDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = TumorClassifier(num_classes=len(settings.tumor_classes)).to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(settings.device), labels.to(settings.device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"epoch {epoch + 1}: avg loss = {running_loss / len(loader):.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"saved checkpoint -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory with one subfolder per class")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/classifier.pt")
    args = parser.parse_args()

    train(args.data_root, args.epochs, args.batch_size, args.lr, args.out)
