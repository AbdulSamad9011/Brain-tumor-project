"""Quick classifier smoke test — runs inference on a local test directory.

Usage: python scripts/test_classifier.py --test-dir /path/to/Testing
"""
from __future__ import annotations

import argparse
import os

import torch

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.io import load_image_2d
from brain_tumor_dx.data.preprocessing import preprocess_for_classifier
from brain_tumor_dx.models.classifier import TumorClassifier


def main(test_dir: str, max_per_class: int = 20):
    model = TumorClassifier(num_classes=len(settings.tumor_classes))
    model.load_state_dict(
        torch.load(settings.classifier_ckpt_path, map_location=settings.device)
    )
    model.to(settings.device).eval()

    classes = settings.tumor_classes
    correct = 0
    total = 0
    per_class = {c: {"correct": 0, "total": 0} for c in classes}

    folder_to_class = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "notumor": "no_tumor",
        "pituitary": "pituitary",
    }

    for folder, true_label in folder_to_class.items():
        folder_path = os.path.join(test_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  Skipping {folder_path} (not found)")
            continue
        images = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]
        for fname in images[:max_per_class]:
            path = os.path.join(folder_path, fname)
            img = load_image_2d(path)
            arr = preprocess_for_classifier(img, settings.classifier_input_size)
            tensor = torch.from_numpy(arr).unsqueeze(0).to(settings.device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=-1).squeeze()
            pred = classes[probs.argmax().item()]
            total += 1
            per_class[true_label]["total"] += 1
            if pred == true_label:
                correct += 1
                per_class[true_label]["correct"] += 1

    print("=== Classification Results ===\n")
    for cls in classes:
        c = per_class[cls]
        acc = c["correct"] / c["total"] if c["total"] > 0 else 0
        print(f"  {cls}: {c['correct']}/{c['total']} ({acc:.0%})")

    if total > 0:
        print(f"\nOverall: {correct}/{total} ({correct / total:.0%})")
    else:
        print("\nNo images found to test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier smoke test")
    parser.add_argument("--test-dir", required=True, help="Directory with one subfolder per class")
    parser.add_argument("--max-per-class", type=int, default=20)
    args = parser.parse_args()
    main(args.test_dir, args.max_per_class)
