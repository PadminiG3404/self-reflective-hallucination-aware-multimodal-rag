"""Self-reflective verification engine."""
from __future__ import annotations

from typing import List, Optional, Tuple

from src.utils.schemas import HallucinationReport, ReasoningStep, ReflectionReport


class SelfReflectiveVerifier:
    def __init__(
        self,
        max_iterations: int = 2,
        weak_step_threshold: float = 0.35,
        missing_evidence_threshold: float = 0.2,
    ) -> None:
        self.max_iterations = max_iterations
        self.weak_step_threshold = weak_step_threshold
        self.missing_evidence_threshold = missing_evidence_threshold

    def verify(
        self, steps: List[ReasoningStep], hallucination_report: HallucinationReport
    ) -> List[ReasoningStep]:
        revised_steps, _ = self.reflect(steps, hallucination_report)
        return revised_steps

    def reflect(
        self,
        steps: List[ReasoningStep],
        hallucination_report: HallucinationReport,
        revised_query: Optional[str] = None,
        prior_hallucination_score: Optional[float] = None,
        retrieval_improvement: Optional[float] = None,
    ) -> Tuple[List[ReasoningStep], ReflectionReport]:
        revised_steps = steps
        invalid_steps, missing_evidence = self.critique_reasoning(revised_steps, hallucination_report)
        for _ in range(self.max_iterations):
            if not invalid_steps:
                break
            revised_steps = self.revise_reasoning(revised_steps, invalid_steps)
            invalid_steps, missing_evidence = self.critique_reasoning(revised_steps, hallucination_report)
        confidence_improvement = self._confidence_delta(steps, revised_steps)
        hallucination_reduction = 0.0
        if prior_hallucination_score is not None:
            hallucination_reduction = max(0.0, prior_hallucination_score - hallucination_report.hallucination_score)
        report = ReflectionReport(
            invalid_steps=invalid_steps,
            missing_evidence=missing_evidence,
            revised_query=revised_query,
            confidence_improvement=confidence_improvement,
            hallucination_reduction=hallucination_reduction,
            retrieval_improvement=retrieval_improvement or 0.0,
        )
        return revised_steps, report

    def critique_reasoning(
        self, steps: List[ReasoningStep], hallucination_report: HallucinationReport
    ) -> Tuple[List[str], List[str]]:
        invalid_steps = [
            step.step_id for step in steps if step.confidence < self.weak_step_threshold
        ]
        missing_evidence = [
            step.step_id
            for step in steps
            if not step.evidence_ids or step.confidence < self.missing_evidence_threshold
        ]
        if hallucination_report.hallucination_score == 0.0:
            return invalid_steps, missing_evidence
        affected = set(hallucination_report.affected_nodes)
        invalid_steps = list(set(invalid_steps) | affected)
        return invalid_steps, missing_evidence

    def revise_reasoning(self, steps: List[ReasoningStep], critique: List[str]) -> List[ReasoningStep]:
        revised = []
        for step in steps:
            if step.step_id in critique:
                if hasattr(step, "model_copy"):
                    revised.append(step.model_copy(update={"confidence": step.confidence * 0.8}))
                else:
                    revised.append(step.copy(update={"confidence": step.confidence * 0.8}))
            else:
                revised.append(step)
        return revised

    def recommend_corrections(self, steps: List[ReasoningStep]) -> List[str]:
        return [step.step_id for step in steps if step.confidence < 0.4]

    @staticmethod
    def _confidence_delta(before: List[ReasoningStep], after: List[ReasoningStep]) -> float:
        if not before or not after:
            return 0.0
        before_avg = sum(step.confidence for step in before) / len(before)
        after_avg = sum(step.confidence for step in after) / len(after)
        return float(after_avg - before_avg)
