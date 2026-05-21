from src.explainability.explainer import ExplainableGenerator
from src.utils.schemas import EvidenceChunk, HallucinationReport, ReasoningStep, UncertaintyReport


def test_explainable_generator():
    generator = ExplainableGenerator()
    explanation = generator.generate(
        answer="a",
        confidence=0.5,
        evidence_chain=[EvidenceChunk(chunk_id="e1", text="t", source="s", score=0.1)],
        reasoning_steps=[ReasoningStep(step_id="s1", statement="", evidence_ids=[], confidence=0.5)],
        hallucination_report=HallucinationReport(
            hallucination_score=0.0, hallucination_type="none", affected_nodes=[]
        ),
        uncertainty_report=UncertaintyReport(
            predictive_uncertainty=0.1,
            retrieval_uncertainty=0.2,
            reasoning_uncertainty=0.3,
            calibration_score=0.4,
        ),
    )
    assert explanation.answer == "a"
