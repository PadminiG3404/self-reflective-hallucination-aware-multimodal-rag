import time

from src.evaluation.runner import EvaluationRunner
from src.utils.schemas import EvidenceChunk, FinalExplanation, HallucinationReport, ReasoningStep, UncertaintyReport


def test_evaluation_runner_summary_keys():
    runner = EvaluationRunner()
    explanation = FinalExplanation(
        answer="a",
        confidence=0.5,
        evidence_chain=[EvidenceChunk(chunk_id="e1", text="t", source="s", score=0.1)],
        reasoning_steps=[ReasoningStep(step_id="s1", statement="", evidence_ids=[], confidence=0.5)],
        hallucination_report=HallucinationReport(
            hallucination_score=0.2,
            hallucination_type="none",
            affected_nodes=[],
        ),
        uncertainty_report=UncertaintyReport(
            predictive_uncertainty=0.1,
            retrieval_uncertainty=0.2,
            reasoning_uncertainty=0.3,
            calibration_score=0.4,
        ),
    )
    start = time.time()
    end = start + 0.01
    result = runner.evaluate([explanation], [start], [end])
    assert "hallucination_score" in result.summaries
    assert "grounding_score" in result.summaries
    assert "uncertainty" in result.summaries
    assert "latency_ms" in result.summaries
    assert "hallucination_rate" in result.summaries
    assert "ece" in result.summaries


def test_evaluation_runner_failure_analysis(tmp_path):
    config = {
        "evaluation": {
            "enable_failure_analysis": True,
            "failure_output_dir": str(tmp_path),
        }
    }
    runner = EvaluationRunner(config=config)
    explanation = FinalExplanation(
        answer="a",
        confidence=0.2,
        evidence_chain=[EvidenceChunk(chunk_id="e1", text="t", source="s", score=0.1)],
        reasoning_steps=[ReasoningStep(step_id="s1", statement="", evidence_ids=[], confidence=0.2)],
        hallucination_report=HallucinationReport(
            hallucination_score=0.8,
            hallucination_type="retrieval_drift",
            affected_nodes=["s1"],
            dominant_factor="retrieval_drift",
        ),
        uncertainty_report=UncertaintyReport(
            predictive_uncertainty=0.1,
            retrieval_uncertainty=0.2,
            reasoning_uncertainty=0.3,
            calibration_score=0.4,
        ),
    )
    start = time.time()
    end = start + 0.01
    result = runner.evaluate([explanation], [start], [end], metadata=[{"query": "q", "image_id": "1"}])
    assert result.failure_records is not None
    assert result.failure_summary is not None


def test_evaluation_runner_recall_with_labels():
    runner = EvaluationRunner()
    explanation = FinalExplanation(
        answer="a",
        confidence=0.5,
        evidence_chain=[EvidenceChunk(chunk_id="e1", text="t", source="s", score=0.1)],
        reasoning_steps=[ReasoningStep(step_id="s1", statement="", evidence_ids=[], confidence=0.5)],
        hallucination_report=HallucinationReport(
            hallucination_score=0.2,
            hallucination_type="none",
            affected_nodes=[],
        ),
        uncertainty_report=UncertaintyReport(
            predictive_uncertainty=0.1,
            retrieval_uncertainty=0.2,
            reasoning_uncertainty=0.3,
            calibration_score=0.4,
        ),
    )
    start = time.time()
    end = start + 0.01
    result = runner.evaluate(
        [explanation],
        [start],
        [end],
        metadata=[{"relevant_chunk_ids": ["e1", "e2"]}],
    )
    assert result.per_example[0]["retrieval_recall_at_k"] == 0.5
