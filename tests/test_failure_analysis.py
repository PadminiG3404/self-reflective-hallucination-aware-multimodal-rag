from src.evaluation.failure_analysis import FailureAnalyzer
from src.utils.schemas import (
    EvidenceChunk,
    FinalExplanation,
    HallucinationReport,
    ReasoningStep,
    UncertaintyReport,
)


def test_failure_analysis_categorizes():
    analyzer = FailureAnalyzer()
    explanation = FinalExplanation(
        answer="a",
        confidence=0.2,
        evidence_chain=[EvidenceChunk(chunk_id="e1", text="t", source="s", score=0.1)],
        reasoning_steps=[ReasoningStep(step_id="s1", statement="step", evidence_ids=[], confidence=0.2)],
        hallucination_report=HallucinationReport(
            hallucination_score=0.8,
            hallucination_type="retrieval_drift",
            affected_nodes=["s1"],
            dominant_factor="retrieval_drift",
        ),
        uncertainty_report=UncertaintyReport(
            predictive_uncertainty=0.2,
            retrieval_uncertainty=0.3,
            reasoning_uncertainty=0.4,
            calibration_score=0.2,
        ),
        trace_summary="step",
    )
    records = analyzer.analyze([explanation])
    assert records
    assert records[0].predicted_failure_type == "retrieval_drift"
