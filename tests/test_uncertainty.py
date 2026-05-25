import numpy as np

from src.uncertainty.estimator import UncertaintyEstimator


def test_uncertainty_estimation():
    estimator = UncertaintyEstimator(temperature=1.0)
    scores = np.array([0.2, 0.3, 0.5])
    report = estimator.estimate(scores, retrieval_confidence=0.6, reasoning_confidence=0.4)
    assert report.calibration_score == 0.5


def test_uncertainty_ece():
    estimator = UncertaintyEstimator(temperature=1.0)
    predicted = np.array([0.2, 0.8, 0.6, 0.4])
    observed = np.array([0.0, 1.0, 1.0, 0.0])
    ece = estimator.estimate_ece(predicted, observed, bins=2)
    assert ece >= 0.0
