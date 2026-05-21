from src.hallucination.detector import HallucinationDetector
from src.utils.schemas import ReasoningStep


def test_hallucination_detection():
    detector = HallucinationDetector(similarity_threshold=0.5, contradiction_threshold=0.5)
    steps = [
        ReasoningStep(step_id="s1", statement="", evidence_ids=[], confidence=0.1),
        ReasoningStep(step_id="s2", statement="", evidence_ids=[], confidence=0.9),
    ]
    report = detector.detect(steps)
    assert report.hallucination_score > 0
    assert "s1" in report.affected_nodes
