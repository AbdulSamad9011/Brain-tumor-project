"""Streamlit front end: upload a scan, see the classification, the
segmentation-derived stats, and the generated report side by side.

Run: streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from brain_tumor_dx.config import settings
from brain_tumor_dx.data.io import load_nifti, load_image_2d
from brain_tumor_dx.data.preprocessing import preprocess_for_classifier
from brain_tumor_dx.models.classifier import TumorClassifier


def enhance_image(image: np.ndarray) -> Image.Image:
    """Percentile windowing + contrast stretch for sharp medical image display."""
    p2, p98 = np.percentile(image, (2, 98))
    windowed = np.clip(image, p2, p98)
    normalized = ((windowed - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(normalized)
    img = img.resize((512, 512), Image.LANCZOS)
    return img


st.set_page_config(page_title="Brain Tumor Diagnosis Assistant", layout="wide")
st.title("Brain Tumor Diagnosis Assistant")
st.caption(
    "Decision-support only - every report requires confirmation by a qualified radiologist."
)

uploaded = st.file_uploader(
    "Upload an MRI image (.jpg / .png) or volume (.nii / .nii.gz)",
    type=["jpg", "jpeg", "png", "nii", "gz"],
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()

    if suffix in (".nii", ".gz"):
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        with st.spinner("Loading volume..."):
            volume = load_nifti(tmp_path)
            mid_slice = volume[volume.shape[0] // 2]

        tmp_path.unlink(missing_ok=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Mid-axial slice")
            st.image(enhance_image(mid_slice), use_container_width=True)

    else:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        image = load_image_2d(tmp_path)
        tmp_path.unlink(missing_ok=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Uploaded image")
            st.image(enhance_image(image), use_container_width=True)

    # --- Classification ---
    with st.spinner("Classifying..."):
        model = TumorClassifier(num_classes=len(settings.tumor_classes))
        model.load_state_dict(
            __import__("torch").load(settings.classifier_ckpt_path, map_location=settings.device)
        )
        model.to(settings.device).eval()

        if suffix in (".nii", ".gz"):
            arr = preprocess_for_classifier(mid_slice, settings.classifier_input_size)
        else:
            arr = preprocess_for_classifier(image, settings.classifier_input_size)

        tensor = __import__("torch").from_numpy(arr).unsqueeze(0).to(settings.device)
        with __import__("torch").no_grad():
            probs = __import__("torch").softmax(model(tensor), dim=-1).squeeze().cpu().numpy()

    with col2:
        st.subheader("Classification Results")
        classes = settings.tumor_classes
        pred_idx = int(probs.argmax())
        st.metric("Predicted type", classes[pred_idx], f"{probs[pred_idx]:.0%} confidence")
        prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        st.bar_chart(prob_dict)

    st.subheader("All probabilities")
    for cls, p in zip(classes, probs):
        st.write(f"**{cls}**: {p:.4f}")
