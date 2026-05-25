"""Explainable response generator."""
from __future__ import annotations

from typing import List

from src.utils.schemas import (
    EvidenceChunk,
    FinalExplanation,
    HallucinationReport,
    ReasoningStep,
    ReflectionReport,
    RetrievalMetrics,
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
        retrieval_metrics: RetrievalMetrics | None = None,
        reflection_report: ReflectionReport | None = None,
        trace_summary: str | None = None,
    ) -> FinalExplanation:
        return FinalExplanation(
            answer=answer,
            confidence=confidence,
            evidence_chain=evidence_chain,
            reasoning_steps=reasoning_steps,
            hallucination_report=hallucination_report,
            uncertainty_report=uncertainty_report,
            retrieval_metrics=retrieval_metrics,
            reflection_report=reflection_report,
            trace_summary=trace_summary,
        )

    def build_trace(self, reasoning_steps: List[ReasoningStep]) -> str:
        return " -> ".join(step.statement for step in reasoning_steps)

    def summarize_evidence(self, evidence_chain: List[EvidenceChunk]) -> str:
        return "; ".join(chunk.text for chunk in evidence_chain)

    def generate_answer_from_evidence(
        self, query: str, evidence_chain: List[EvidenceChunk], max_chars: int = 280
    ) -> str:
        if not evidence_chain:
            return f"No supporting evidence found for: {query}"
        top_text = evidence_chain[0].text.strip()
        if not top_text:
            return f"Insufficient evidence for: {query}"
        return self._truncate_text(top_text, max_chars)

    def build_answer_prompt(self, query: str, evidence_chain: List[EvidenceChunk]) -> str:
        evidence_text = " ".join(chunk.text for chunk in evidence_chain[:3])
        return (
            "Answer the question using only the evidence. "
            "If evidence is insufficient, say you do not know.\n"
            f"Question: {query}\n"
            f"Evidence: {evidence_text}\n"
            "Answer:"
        )

    def generate_answer_with_seq2seq_model(
        self,
        query: str,
        evidence_chain: List[EvidenceChunk],
        model,
        tokenizer,
        device,
        max_new_tokens: int = 96,
    ) -> str:
        if model is None or tokenizer is None:
            return self.generate_answer_from_evidence(query, evidence_chain)
        prompt = self.build_answer_prompt(query, evidence_chain)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if outputs is None or outputs.shape[0] == 0:
            return self.generate_answer_from_evidence(query, evidence_chain)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if not text:
            return self.generate_answer_from_evidence(query, evidence_chain)
        return text

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        trimmed = text[: max_chars - 3].rstrip()
        return f"{trimmed}..."
