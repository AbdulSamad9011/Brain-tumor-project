"""Central configuration for the brain tumor diagnosis pipeline.

Mirrors the settings pattern from the research-agent project: everything
is env-driven with sane defaults, loaded once via a module-level Settings
instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _resolve_device() -> str:
    import torch

    requested = os.getenv("DEVICE", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()[0]
        if cap < 7:
            print(
                f"[config] GPU compute capability {cap}.x — older GPU, training may be slower."
            )
        return "cuda"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[config] CUDA requested but not available — falling back to CPU.")
        return "cpu"
    return requested


@dataclass
class Settings:
    # API keys (report generation only — never touch pixel-level inference)
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Checkpoints
    classifier_ckpt_path: Path = field(
        default_factory=lambda: Path(_env("CLASSIFIER_CKPT_PATH", "checkpoints/classifier.pt"))
    )
    segmentation_ckpt_path: Path = field(
        default_factory=lambda: Path(_env("SEGMENTATION_CKPT_PATH", "checkpoints/segmentation.pt"))
    )

    # Inference
    device: str = field(default_factory=_resolve_device)
    classifier_input_size: int = field(default_factory=lambda: int(_env("CLASSIFIER_INPUT_SIZE", "224")))
    segmentation_input_size: int = field(default_factory=lambda: int(_env("SEGMENTATION_INPUT_SIZE", "128")))
    confidence_threshold: float = field(default_factory=lambda: float(_env("CONFIDENCE_THRESHOLD", "0.5")))

    # Report generation
    report_model: str = field(default_factory=lambda: _env("REPORT_MODEL", "groq:openai/gpt-oss-120b"))

    # Class taxonomy — order matters, must match training label encoding
    tumor_classes: tuple[str, ...] = ("glioma", "meningioma", "pituitary", "no_tumor")


settings = Settings()
