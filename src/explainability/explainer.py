"""Explainable response generator."""
from __future__ import annotations

from typing import List

from src.utils.schemas import (
    EvidenceChunk,
    FinalExplanation,
    HallucinationReport,
    ReasoningStep,
    UncertaintyReport,
)


class ExplainableGenerator:
    def generate(
        self,
        answer: str,
        confidence: float,
        evidence_chain: List[EvidenceChunk],
        reasoning_steps: List[ReasoningStep],
        hallucination_report: HallucinationReport,
        uncertainty_report: UncertaintyReport,
        trace_summary: str | None = None,
    ) -> FinalExplanation:
        return FinalExplanation(
            answer=answer,
            confidence=confidence,
            evidence_chain=evidence_chain,
            reasoning_steps=reasoning_steps,
            hallucination_report=hallucination_report,
            uncertainty_report=uncertainty_report,
            trace_summary=trace_summary,
        )

    def build_trace(self, reasoning_steps: List[ReasoningStep]) -> str:
        return " -> ".join(step.statement for step in reasoning_steps)

    def summarize_evidence(self, evidence_chain: List[EvidenceChunk]) -> str:
        return "; ".join(chunk.text for chunk in evidence_chain)
