"""Failure case analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import csv
import json

from src.utils.schemas import FinalExplanation


@dataclass
class FailureRecord:
    query: str
    image_id: Optional[str]
    reasoning_path: str
    hallucination_score: float
    uncertainty_score: float
    predicted_failure_type: str


class FailureAnalyzer:
    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self.thresholds = thresholds or {
            "high_hallucination": 0.6,
            "low_confidence": 0.4,
            "low_evidence": 0.2,
        }

    def analyze(
        self,
        explanations: Iterable[FinalExplanation],
        metadata: Optional[List[Dict[str, str]]] = None,
    ) -> List[FailureRecord]:
        records: List[FailureRecord] = []
        meta_list = metadata or []
        for idx, explanation in enumerate(explanations):
            meta = meta_list[idx] if idx < len(meta_list) else {}
            query = str(meta.get("query", ""))
            image_id = meta.get("image_id")
            reasoning_path = explanation.trace_summary or " -> ".join(
                step.statement for step in explanation.reasoning_steps
            )
            hallucination_score = explanation.hallucination_report.hallucination_score
            uncertainty_score = explanation.uncertainty_report.calibration_score
            failure_type = self.categorize(explanation)
            records.append(
                FailureRecord(
                    query=query,
                    image_id=image_id,
                    reasoning_path=reasoning_path,
                    hallucination_score=hallucination_score,
                    uncertainty_score=uncertainty_score,
                    predicted_failure_type=failure_type,
                )
            )
        return records

    def categorize(self, explanation: FinalExplanation) -> str:
        report = explanation.hallucination_report
        dominant = report.dominant_factor or report.hallucination_type
        if dominant in {"retrieval_drift", "retrieval"}:
            return "retrieval_drift"
        if dominant == "graph_inconsistency":
            return "graph_inconsistency"
        if dominant == "contradiction":
            return "contradiction_miss"
        if explanation.reflection_report and explanation.reflection_report.confidence_improvement <= 0.0:
            return "reflection_failure"
        evidence_scores = [chunk.score for chunk in explanation.evidence_chain]
        if evidence_scores and max(evidence_scores) < self.thresholds["low_evidence"]:
            return "low_evidence_coverage"
        if (
            explanation.uncertainty_report.calibration_score < self.thresholds["low_confidence"]
            and report.hallucination_score >= self.thresholds["high_hallucination"]
        ):
            return "uncertainty_overconfidence"
        if report.hallucination_score >= self.thresholds["high_hallucination"]:
            return "visual_ambiguity"
        return "none"

    def export(self, records: List[FailureRecord], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "failure_analysis.json"
        csv_path = output_dir / "failure_analysis.csv"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump([record.__dict__ for record in records], handle, indent=2)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FailureRecord.__annotations__.keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record.__dict__)
