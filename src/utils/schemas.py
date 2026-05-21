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
    notes: Optional[str] = None


class UncertaintyReport(BaseModel):
    predictive_uncertainty: float
    retrieval_uncertainty: float
    reasoning_uncertainty: float
    calibration_score: float


class FinalExplanation(BaseModel):
    answer: str
    confidence: float
    evidence_chain: List[EvidenceChunk]
    reasoning_steps: List[ReasoningStep]
    hallucination_report: HallucinationReport
    uncertainty_report: UncertaintyReport
    trace_summary: Optional[str] = None
