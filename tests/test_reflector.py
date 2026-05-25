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


def test_reflector_report_fields():
    verifier = SelfReflectiveVerifier(max_iterations=1)
    steps = [ReasoningStep(step_id="s1", statement="", evidence_ids=[], confidence=0.2)]
    report = HallucinationReport(
        hallucination_score=0.6,
        hallucination_type="weak_grounding",
        affected_nodes=["s1"],
    )
    revised_steps, reflection = verifier.reflect(steps, report, revised_query="focus")
    assert revised_steps
    assert reflection.revised_query == "focus"
    assert "s1" in reflection.invalid_steps
