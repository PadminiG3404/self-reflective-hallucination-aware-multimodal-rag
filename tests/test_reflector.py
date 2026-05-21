from src.reflection.self_reflector import SelfReflectiveVerifier
from src.utils.schemas import HallucinationReport, ReasoningStep


def test_reflector_reduces_confidence():
    verifier = SelfReflectiveVerifier(max_iterations=1)
    steps = [ReasoningStep(step_id="s1", statement="", evidence_ids=[], confidence=0.3)]
    report = HallucinationReport(
        hallucination_score=0.5,
        hallucination_type="weak_grounding",
        affected_nodes=["s1"],
    )
    revised = verifier.verify(steps, report)
    assert revised[0].confidence < steps[0].confidence
