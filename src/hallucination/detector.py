"""Hallucination detection heuristics."""
from __future__ import annotations

from typing import Callable, List, Optional

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
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.nli_pipeline = nli_pipeline

    def detect(
        self,
        steps: List[ReasoningStep],
        retrieval_metrics: RetrievalMetrics | None = None,
        graph_consistency: float | None = None,
        evidence_texts: List[str] | None = None,
    ) -> HallucinationReport:
        weak_steps = [step for step in steps if step.confidence < self.similarity_threshold]
        step_score = float(len(weak_steps) / max(1, len(steps)))
        retrieval_score = 0.0
        if retrieval_metrics is not None:
            retrieval_score = 1.0 - retrieval_metrics.evidence_confidence
        graph_score = 0.0 if graph_consistency is None else float(max(0.0, 1.0 - graph_consistency))
        contradiction_score = 0.0
        if self.nli_pipeline is not None and evidence_texts:
            contradiction_score = self.contradiction_score(steps, evidence_texts)
        grounding_score = 0.0
        if evidence_texts:
            grounding_score = self.grounding_score(steps, evidence_texts)
        score = float(
            min(1.0, (step_score + retrieval_score + graph_score + contradiction_score + grounding_score) / 5.0)
        )
        if retrieval_score > step_score and retrieval_score > 0.5:
            hallucination_type = "retrieval_drift"
        elif contradiction_score > self.contradiction_threshold:
            hallucination_type = "semantic_contradiction"
        elif grounding_score > 0.5:
            hallucination_type = "weak_grounding"
        elif weak_steps:
            hallucination_type = "weak_grounding"
        else:
            hallucination_type = "none"
        return HallucinationReport(
            hallucination_score=score,
            hallucination_type=hallucination_type,
            affected_nodes=[step.step_id for step in weak_steps],
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
