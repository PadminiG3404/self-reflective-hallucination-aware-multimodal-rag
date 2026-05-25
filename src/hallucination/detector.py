"""Hallucination detection heuristics."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.schemas import HallucinationReport, ReasoningStep, RetrievalMetrics


class HallucinationDetector:
    def __init__(
        self,
        similarity_threshold: float,
        contradiction_threshold: float,
        nli_pipeline: Optional[Callable[[str, str], dict | list]] = None,
        factor_weights: Optional[Dict[str, float]] = None,
        factor_enabled: Optional[Dict[str, bool]] = None,
        severity_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.nli_pipeline = nli_pipeline
        self.factor_weights = factor_weights or {
            "grounding": 0.3,
            "contradiction": 0.3,
            "retrieval_drift": 0.2,
            "graph_inconsistency": 0.2,
        }
        self.factor_enabled = factor_enabled or {
            "grounding": True,
            "contradiction": True,
            "retrieval_drift": True,
            "graph_inconsistency": True,
        }
        self.severity_thresholds = severity_thresholds or {"medium": 0.4, "high": 0.7}

    def detect(
        self,
        steps: List[ReasoningStep],
        retrieval_metrics: RetrievalMetrics | None = None,
        graph_consistency: float | None = None,
        evidence_texts: List[str] | None = None,
    ) -> HallucinationReport:
        weak_steps = [step for step in steps if step.confidence < self.similarity_threshold]
        retrieval_score = self._retrieval_drift_score(retrieval_metrics)
        graph_score = 0.0 if graph_consistency is None else float(max(0.0, 1.0 - graph_consistency))
        contradiction_score = 0.0
        if self.nli_pipeline is not None and evidence_texts:
            contradiction_score = self.contradiction_score(steps, evidence_texts)
        grounding_score = 0.0
        if evidence_texts:
            grounding_score = self.grounding_score(steps, evidence_texts)
        factor_scores = {
            "grounding": grounding_score,
            "contradiction": contradiction_score,
            "retrieval_drift": retrieval_score,
            "graph_inconsistency": graph_score,
        }
        score = self._weighted_score(factor_scores)
        dominant_factor = self._dominant_factor(factor_scores)
        severity_level = self._severity_level(score)
        if score == 0.0:
            hallucination_type = "none"
        else:
            hallucination_type = dominant_factor or "unknown"
        return HallucinationReport(
            hallucination_score=score,
            hallucination_type=hallucination_type,
            affected_nodes=[step.step_id for step in weak_steps],
            factor_scores=factor_scores,
            dominant_factor=dominant_factor,
            severity_level=severity_level,
        )

    def contradiction_score(self, steps: List[ReasoningStep], evidence_texts: List[str]) -> float:
        if not steps or not evidence_texts:
            return 0.0
        evidence_blob = " ".join(evidence_texts)
        scores = []
        for step in steps:
            result = self.nli_pipeline(step.statement, evidence_blob)
            scores.append(self._extract_contradiction_score(result))
        return float(np.mean(scores))

    def grounding_score(self, steps: List[ReasoningStep], evidence_texts: List[str]) -> float:
        if not steps or not evidence_texts:
            return 0.0
        evidence_blob = " ".join(evidence_texts)
        corpus = [evidence_blob] + [step.statement for step in steps]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        evidence_vec = matrix[0:1]
        step_vecs = matrix[1:]
        sims = cosine_similarity(step_vecs, evidence_vec).reshape(-1)
        if sims.size == 0:
            return 0.0
        mean_sim = float(np.mean(sims))
        return float(max(0.0, 1.0 - mean_sim))

    def _retrieval_drift_score(self, retrieval_metrics: RetrievalMetrics | None) -> float:
        if retrieval_metrics is None:
            return 0.0
        confidence_gap = 1.0 - retrieval_metrics.evidence_confidence
        consistency = retrieval_metrics.retrieval_consistency
        return float(max(0.0, min(1.0, 0.5 * confidence_gap + 0.5 * consistency)))

    def _weighted_score(self, factor_scores: Dict[str, float]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for factor, score in factor_scores.items():
            if not self.factor_enabled.get(factor, True):
                continue
            weight = float(self.factor_weights.get(factor, 0.0))
            total_weight += weight
            weighted_sum += weight * score
        if total_weight <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, weighted_sum / total_weight)))

    def _dominant_factor(self, factor_scores: Dict[str, float]) -> Optional[str]:
        candidates = {
            factor: score
            for factor, score in factor_scores.items()
            if self.factor_enabled.get(factor, True)
        }
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def _severity_level(self, score: float) -> str:
        medium = float(self.severity_thresholds.get("medium", 0.4))
        high = float(self.severity_thresholds.get("high", 0.7))
        if score >= high:
            return "high"
        if score >= medium:
            return "medium"
        return "low"

    @staticmethod
    def _extract_contradiction_score(result: dict | list) -> float:
        if isinstance(result, dict):
            label = result.get("label", "").lower()
            score = float(result.get("score", 0.0))
            return score if "contrad" in label else 0.0
        if isinstance(result, list):
            for item in result:
                label = str(item.get("label", "")).lower()
                if "contrad" in label:
                    return float(item.get("score", 0.0))
        return 0.0

    def claim_evidence_consistency(self, scores: List[float]) -> float:
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def graph_consistency_score(self, node_scores: List[float]) -> float:
        if not node_scores:
            return 0.0
        return float(np.mean(node_scores))
