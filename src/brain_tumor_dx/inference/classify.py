"""Run the classification model on a single preprocessed slice."""
from __future__ import annotations

import numpy as np
import torch

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.preprocessing import preprocess_for_classifier
from brain_tumor_dx.models.registry import load_classifier


def classify_slice(image: np.ndarray) -> dict:
    """Returns {"label": str, "confidence": float, "probabilities": dict[str, float]}."""
    model = load_classifier()
    array = preprocess_for_classifier(image, settings.classifier_input_size)
    tensor = torch.from_numpy(array).unsqueeze(0).to(settings.device)  # add batch dim

    probs = model.predict_proba(tensor).squeeze(0).cpu().numpy()
    label_idx = int(np.argmax(probs))

    return {
        "label": settings.tumor_classes[label_idx],
        "confidence": float(probs[label_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(settings.tumor_classes, probs)},
    }
