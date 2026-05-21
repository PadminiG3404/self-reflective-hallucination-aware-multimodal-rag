"""Self-reflective verification engine."""
from __future__ import annotations

from typing import List

from src.utils.schemas import HallucinationReport, ReasoningStep


class SelfReflectiveVerifier:
    def __init__(self, max_iterations: int = 2) -> None:
        self.max_iterations = max_iterations

    def verify(
        self, steps: List[ReasoningStep], hallucination_report: HallucinationReport
    ) -> List[ReasoningStep]:
        revised_steps = steps
        for _ in range(self.max_iterations):
            critique = self.critique_reasoning(revised_steps, hallucination_report)
            if not critique:
                break
            revised_steps = self.revise_reasoning(revised_steps, critique)
        return revised_steps

    def critique_reasoning(
        self, steps: List[ReasoningStep], hallucination_report: HallucinationReport
    ) -> List[str]:
        if hallucination_report.hallucination_score == 0.0:
            return []
        return hallucination_report.affected_nodes

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
