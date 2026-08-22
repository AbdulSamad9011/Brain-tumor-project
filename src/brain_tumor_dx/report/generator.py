"""Turns a DiagnosticFinding into a clinician-readable report.

Deliberately narrow: the LLM only ever sees the structured numbers below,
never the image itself — it narrates, it does not diagnose from pixels.
Follows the same init_chat_model + structured-output pattern as
agents/summarizer.py in the research-agent project.
"""
from __future__ import annotations

from langchain.chat_models import init_chat_model

from brain_tumor_dx.config import settings
from brain_tumor_dx.report.schema import DiagnosticFinding, DiagnosticReport

SYSTEM_PROMPT = """\
You are a radiology report assistant. You will be given structured, \
pre-computed model outputs (tumor type, confidence, volume, location) — \
never raw imaging data. Write a clear, cautious report for clinician review.

Rules:
- Never state a diagnosis as certain; always frame it as a model-generated \
  finding that requires expert confirmation.
- If classification_confidence is below {threshold}, say so explicitly and \
  recommend radiologist review before any conclusion is drawn.
- Do not invent measurements or findings beyond what's provided.
- This report supports a clinician's decision; it does not replace one.
"""


def generate_report(finding: DiagnosticFinding) -> DiagnosticReport:
    model = init_chat_model(settings.report_model).with_structured_output(
        DiagnosticReport, method="json_schema"
    )

    system = SYSTEM_PROMPT.format(threshold=settings.confidence_threshold)
    user = finding.model_dump_json(indent=2)

    return model.invoke([("system", system), ("user", user)])
