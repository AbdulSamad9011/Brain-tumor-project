from __future__ import annotations

from pydantic import BaseModel

from brain_tumor_dx.report.schema import DiagnosticReport


class PredictResponse(BaseModel):
    report: DiagnosticReport
