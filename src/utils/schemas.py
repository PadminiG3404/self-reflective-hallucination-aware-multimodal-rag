"""Pydantic models shared across pipeline stages."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvidenceChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningStep(BaseModel):
    step_id: str
    statement: str
    evidence_ids: List[str]
    confidence: float
    dependencies: List[str] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    semantic_similarity: float
    retrieval_consistency: float
    evidence_confidence: float
    alignment_score: float


class HallucinationReport(BaseModel):
    hallucination_score: float
    hallucination_type: str
    affected_nodes: List[str]
    factor_scores: Dict[str, float] = Field(default_factory=dict)
    dominant_factor: Optional[str] = None
    severity_level: Optional[str] = None
    notes: Optional[str] = None


class UncertaintyReport(BaseModel):
    predictive_uncertainty: float
    retrieval_uncertainty: float
    reasoning_uncertainty: float
    calibration_score: float
    ece: Optional[float] = None
    miscalibration_score: Optional[float] = None


class ReflectionReport(BaseModel):
    invalid_steps: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    revised_query: Optional[str] = None
    confidence_improvement: float = 0.0
    hallucination_reduction: float = 0.0
    retrieval_improvement: float = 0.0


class FinalExplanation(BaseModel):
    answer: str
    confidence: float
    evidence_chain: List[EvidenceChunk]
    reasoning_steps: List[ReasoningStep]
    hallucination_report: HallucinationReport
    uncertainty_report: UncertaintyReport
    retrieval_metrics: Optional[RetrievalMetrics] = None
    reflection_report: Optional[ReflectionReport] = None
    trace_summary: Optional[str] = None
