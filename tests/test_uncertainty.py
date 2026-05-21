import numpy as np

from src.uncertainty.estimator import UncertaintyEstimator


def test_uncertainty_estimation():
    estimator = UncertaintyEstimator(temperature=1.0)
    scores = np.array([0.2, 0.3, 0.5])
    report = estimator.estimate(scores, retrieval_confidence=0.6, reasoning_confidence=0.4)
    assert report.calibration_score == 0.5
