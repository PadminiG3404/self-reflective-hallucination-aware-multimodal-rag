"""Evaluation harness for the multimodal RAG pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from src.evaluation import metrics
from src.utils.schemas import FinalExplanation


@dataclass
class EvaluationResult:
    summaries: Dict[str, float]
    per_example: List[Dict[str, float]]


class EvaluationRunner:
    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        explanations: List[FinalExplanation],
        start_times: List[float],
        end_times: List[float],
    ) -> EvaluationResult:
        per_example = []
        for explanation, start_time, end_time in zip(explanations, start_times, end_times):
            hallucination_score = explanation.hallucination_report.hallucination_score
            grounding_score = 1.0 - hallucination_score
            latency_value = metrics.latency_ms(start_time, end_time)
            per_example.append(
                {
                    "hallucination_score": hallucination_score,
                    "grounding_score": grounding_score,
                    "uncertainty": explanation.uncertainty_report.calibration_score,
                    "latency_ms": latency_value,
                }
            )
        summaries = self._aggregate(per_example)
        return EvaluationResult(summaries=summaries, per_example=per_example)

    def _aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        keys = results[0].keys()
        summary: Dict[str, float] = {}
        for key in keys:
            summary[key] = sum(item[key] for item in results) / len(results)
        return summary
