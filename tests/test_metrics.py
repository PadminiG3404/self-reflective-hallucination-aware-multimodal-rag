import pytest

from src.evaluation import metrics


def test_metrics_functions():
    assert metrics.retrieval_precision(2, 4) == 0.5
    assert metrics.retrieval_recall(2, 4) == 0.5
    assert metrics.hallucination_reduction_rate(0.8, 0.3) == 0.5
    assert metrics.confidence_gain(0.7, 0.4) == pytest.approx(0.3)
