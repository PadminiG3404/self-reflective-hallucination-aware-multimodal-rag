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


def exact_match(predicted: str, references: List[str]) -> float:
    if not references:
        return 0.0
    normalized = _normalize_text(predicted)
    for reference in references:
        if normalized == _normalize_text(reference):
            return 1.0
    return 0.0


def token_f1(predicted: str, references: List[str]) -> tuple[float, float, float]:
    if not references:
        return 0.0, 0.0, 0.0
    pred_tokens = _normalize_text(predicted).split()
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    pred_counts = {token: pred_tokens.count(token) for token in pred_tokens}
    for reference in references:
        ref_tokens = _normalize_text(reference).split()
        if not ref_tokens:
            continue
        ref_counts = {token: ref_tokens.count(token) for token in ref_tokens}
        overlap = 0
        for token, count in pred_counts.items():
            overlap += min(count, ref_counts.get(token, 0))
        precision = overlap / max(1, len(pred_tokens))
        recall = overlap / max(1, len(ref_tokens))
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
    return float(best_precision), float(best_recall), float(best_f1)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return float(1.0 / idx)
    return 0.0


def _normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())
