"""Structured schemas for the diagnosis pipeline — mirrors the
Pydantic-everywhere pattern from the research-agent project so the LLM
report step consumes typed data, never raw model internals."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DiagnosticFinding(BaseModel):
    tumor_present: bool
    tumor_type: str = Field(description="One of the configured tumor classes, or 'no_tumor'")
    classification_confidence: float = Field(ge=0.0, le=1.0)
    class_probabilities: dict[str, float]
    tumor_volume_mm3: float
    tumor_centroid: Optional[tuple[float, float, float]] = None


class DiagnosticReport(BaseModel):
    summary: str = Field(description="1-2 sentence plain-language summary")
    findings: DiagnosticFinding
    narrative: str = Field(description="Full report body, markdown")
    confidence_note: str = Field(description="Explicit caveat about model confidence / limitations")
    recommendation: str = Field(description="Suggested next step, e.g. 'refer for radiologist review'")
