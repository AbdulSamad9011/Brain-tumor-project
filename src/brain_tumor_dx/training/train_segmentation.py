"""Trains TumorSegmenter on paired image/mask volumes (BraTS-style layout).

Uses MONAI's Dice loss, the standard choice for segmentation with heavy
class imbalance (tumor voxels are a tiny fraction of the volume).
"""
from __future__ import annotations

import argparse

import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.datasets import SegmentationDataset
from brain_tumor_dx.models.segmentation import TumorSegmenter


def train(data_root: str, epochs: int, batch_size: int, lr: float, out_path: str):
    dataset = SegmentationDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = TumorSegmenter().to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = DiceLoss(sigmoid=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            images, masks = images.to(settings.device), masks.to(settings.device)

            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"epoch {epoch + 1}: avg dice loss = {running_loss / len(loader):.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"saved checkpoint -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Directory of case_XXX/{image,mask}.nii.gz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="checkpoints/segmentation.pt")
    args = parser.parse_args()

    train(args.data_root, args.epochs, args.batch_size, args.lr, args.out)
