"""Uncertainty estimation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

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
        predicted_confidences: Optional[Iterable[float]] = None,
        observed_outcomes: Optional[Iterable[float]] = None,
        calibration_bins: int = 10,
    ) -> UncertaintyReport:
        retrieval_confidence = self._clamp_confidence(retrieval_confidence)
        reasoning_confidence = self._clamp_confidence(reasoning_confidence)
        predictive_uncertainty = self.entropy_score(predictive_scores)
        retrieval_uncertainty = 1.0 - retrieval_confidence
        reasoning_uncertainty = 1.0 - reasoning_confidence
        calibration_score = self.calibrate_confidence(retrieval_confidence, reasoning_confidence)
        ece = None
        miscalibration = None
        if predicted_confidences is not None and observed_outcomes is not None:
            predicted_arr = np.array(list(predicted_confidences), dtype=np.float32)
            observed_arr = np.array(list(observed_outcomes), dtype=np.float32)
            ece = self.estimate_ece(predicted_arr, observed_arr, bins=calibration_bins)
            miscalibration = self.miscalibration_score(predicted_arr, observed_arr)
        return UncertaintyReport(
            predictive_uncertainty=predictive_uncertainty,
            retrieval_uncertainty=retrieval_uncertainty,
            reasoning_uncertainty=reasoning_uncertainty,
            calibration_score=calibration_score,
            ece=ece,
            miscalibration_score=miscalibration,
        )

    def entropy_score(self, scores: Optional[np.ndarray]) -> float:
        if scores is None or scores.size == 0:
            return 0.0
        scores = scores / self.temperature
        scores = np.clip(scores, 1e-8, 1.0)
        return float(-np.sum(scores * np.log(scores)))

    def calibrate_confidence(self, retrieval_confidence: float, reasoning_confidence: float) -> float:
        return float((retrieval_confidence + reasoning_confidence) / 2.0)

    def estimate_ece(self, predicted: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
        curve = self.calibration_curve(predicted, observed, bins)
        if not curve:
            return 0.0
        return float(sum(abs(acc - conf) * weight for acc, conf, weight in curve))

    def calibration_curve(
        self, predicted: np.ndarray, observed: np.ndarray, bins: int = 10
    ) -> list[Tuple[float, float, float]]:
        if predicted.size == 0 or observed.size == 0:
            return []
        bins = max(1, int(bins))
        predicted = np.clip(predicted, 0.0, 1.0)
        observed = np.clip(observed, 0.0, 1.0)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        curve = []
        for idx in range(bins):
            lower = bin_edges[idx]
            upper = bin_edges[idx + 1]
            mask = (predicted >= lower) & (predicted < upper)
            if idx == bins - 1:
                mask = (predicted >= lower) & (predicted <= upper)
            if not np.any(mask):
                continue
            bin_confidence = float(np.mean(predicted[mask]))
            bin_accuracy = float(np.mean(observed[mask]))
            weight = float(np.mean(mask))
            curve.append((bin_accuracy, bin_confidence, weight))
        return curve

    def miscalibration_score(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        if predicted.size == 0 or observed.size == 0:
            return 0.0
        predicted = np.clip(predicted, 0.0, 1.0)
        observed = np.clip(observed, 0.0, 1.0)
        return float(np.mean(np.abs(predicted - observed)))

    @staticmethod
    def _clamp_confidence(value: float) -> float:
        return float(max(0.0, min(1.0, value)))
