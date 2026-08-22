"""Streamlit front end: upload a scan, see the classification, the
segmentation-derived stats, and the generated report side by side.

Run: streamlit run app.py
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import streamlit as st

from brain_tumor_dx.data.io import load_nifti
from brain_tumor_dx.pipeline import run_pipeline

st.set_page_config(page_title="Brain Tumor Diagnosis Assistant", layout="wide")
st.title("🧠 Brain Tumor Diagnosis Assistant")
st.caption(
    "Decision-support only — every report requires confirmation by a qualified radiologist."
)

uploaded = st.file_uploader("Upload an MRI volume (.nii / .nii.gz)", type=["nii", "gz"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    with st.spinner("Running classification + segmentation..."):
        volume = load_nifti(tmp_path)
        mid_slice = volume[volume.shape[0] // 2]
        report = asyncio.run(run_pipeline(slice_2d=mid_slice, volume_3d=volume))

    tmp_path.unlink(missing_ok=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Mid-axial slice")
        st.image(mid_slice, clamp=True, use_container_width=True)

    with col2:
        st.subheader("Findings")
        f = report.findings
        st.metric("Predicted type", f.tumor_type, f"{f.classification_confidence:.0%} confidence")
        st.metric("Estimated volume", f"{f.tumor_volume_mm3:.1f} mm³")
        st.bar_chart(f.class_probabilities)

    st.subheader("Report")
    st.markdown(report.narrative)
    st.warning(report.confidence_note)
    st.info(f"**Recommendation:** {report.recommendation}")
