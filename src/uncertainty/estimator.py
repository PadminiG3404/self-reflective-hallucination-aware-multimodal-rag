"""Uncertainty estimation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.schemas import UncertaintyReport


class UncertaintyEstimator:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def estimate(
        self,
        predictive_scores: Optional[np.ndarray] = None,
        retrieval_confidence: float = 0.0,
        reasoning_confidence: float = 0.0,
    ) -> UncertaintyReport:
        retrieval_confidence = self._clamp_confidence(retrieval_confidence)
        reasoning_confidence = self._clamp_confidence(reasoning_confidence)
        predictive_uncertainty = self.entropy_score(predictive_scores)
        retrieval_uncertainty = 1.0 - retrieval_confidence
        reasoning_uncertainty = 1.0 - reasoning_confidence
        calibration_score = self.calibrate_confidence(retrieval_confidence, reasoning_confidence)
        return UncertaintyReport(
            predictive_uncertainty=predictive_uncertainty,
            retrieval_uncertainty=retrieval_uncertainty,
            reasoning_uncertainty=reasoning_uncertainty,
            calibration_score=calibration_score,
        )

    def entropy_score(self, scores: Optional[np.ndarray]) -> float:
        if scores is None or scores.size == 0:
            return 0.0
        scores = scores / self.temperature
        scores = np.clip(scores, 1e-8, 1.0)
        return float(-np.sum(scores * np.log(scores)))

    def calibrate_confidence(self, retrieval_confidence: float, reasoning_confidence: float) -> float:
        return float((retrieval_confidence + reasoning_confidence) / 2.0)

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return float(max(0.0, min(1.0, value)))
