"""Evaluation metrics for hallucination-aware RAG."""
from __future__ import annotations

from typing import List

import numpy as np


def hallucination_reduction_rate(before: float, after: float) -> float:
    return float(max(0.0, before - after))


def hallucination_rate(scores: List[float], threshold: float = 0.5) -> float:
    if not scores:
        return 0.0
    hits = sum(1 for score in scores if score >= threshold)
    return float(hits / len(scores))


def retrieval_precision(relevant: int, retrieved: int) -> float:
    if retrieved == 0:
        return 0.0
    return float(relevant / retrieved)


def retrieval_recall(relevant: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return float(relevant / total_relevant)


def retrieval_recall_at_k(relevant_at_k: int, total_relevant: int) -> float:
    return retrieval_recall(relevant_at_k, total_relevant)


def reasoning_consistency(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return float(np.mean(scores))


def factual_grounding_score(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return float(np.mean(scores))


def reasoning_faithfulness(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return float(np.mean(scores))


def uncertainty_calibration_error(predicted: List[float], observed: List[float]) -> float:
    if not predicted or not observed:
        return 0.0
    predicted_arr = np.array(predicted)
    observed_arr = np.array(observed)
    return float(np.mean(np.abs(predicted_arr - observed_arr)))


def confidence_gain(after: float, before: float) -> float:
    return float(after - before)


def explainability_completeness(trace: str, evidence_count: int) -> float:
    if not trace:
        return 0.0
    return float(min(1.0, len(trace) / max(1, evidence_count)))


def latency_ms(start_time: float, end_time: float) -> float:
    return float(max(0.0, (end_time - start_time) * 1000.0))
