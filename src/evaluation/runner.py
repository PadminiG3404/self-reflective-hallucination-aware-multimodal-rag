"""Evaluation harness for the multimodal RAG pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.evaluation import metrics
from src.evaluation.failure_analysis import FailureAnalyzer
from src.uncertainty.estimator import UncertaintyEstimator
from src.utils.schemas import FinalExplanation


@dataclass
class EvaluationResult:
    summaries: Dict[str, float]
    per_example: List[Dict[str, float]]
    failure_records: Optional[List[Dict[str, object]]] = None
    failure_summary: Optional[Dict[str, float]] = None


class EvaluationRunner:
    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    def evaluate(
        self,
        explanations: List[FinalExplanation],
        start_times: List[float],
        end_times: List[float],
        metadata: Optional[List[Dict[str, object]]] = None,
    ) -> EvaluationResult:
        per_example = []
        hallucination_scores = []
        predicted_confidences = []
        observed_outcomes = []
        for idx, (explanation, start_time, end_time) in enumerate(
            zip(explanations, start_times, end_times)
        ):
            hallucination_score = explanation.hallucination_report.hallucination_score
            grounding_score = 1.0 - hallucination_score
            latency_value = metrics.latency_ms(start_time, end_time)
            retrieval_metrics = explanation.retrieval_metrics
            retrieval_precision_proxy = (
                retrieval_metrics.semantic_similarity if retrieval_metrics is not None else 0.0
            )
            retrieval_recall_proxy = (
                retrieval_metrics.evidence_confidence if retrieval_metrics is not None else 0.0
            )
            retrieval_precision = retrieval_precision_proxy
            retrieval_recall_at_k = retrieval_recall_proxy
            if metadata and idx < len(metadata):
                relevant_ids = metadata[idx].get("relevant_chunk_ids")
                if isinstance(relevant_ids, list) and relevant_ids:
                    retrieved_ids = [chunk.chunk_id for chunk in explanation.evidence_chain]
                    hits = sum(1 for chunk_id in retrieved_ids if chunk_id in relevant_ids)
                    retrieval_precision = metrics.retrieval_precision(hits, len(retrieved_ids))
                    retrieval_recall_at_k = metrics.retrieval_recall(hits, len(relevant_ids))
            reasoning_faithfulness = metrics.reasoning_faithfulness(
                [step.confidence for step in explanation.reasoning_steps]
            )
            reflection_gain = 0.0
            hallucination_reduction = 0.0
            if explanation.reflection_report is not None:
                reflection_gain = explanation.reflection_report.confidence_improvement
                hallucination_reduction = explanation.reflection_report.hallucination_reduction
            per_example.append(
                {
                    "hallucination_score": hallucination_score,
                    "grounding_score": grounding_score,
                    "uncertainty": explanation.uncertainty_report.calibration_score,
                    "latency_ms": latency_value,
                    "reasoning_faithfulness": reasoning_faithfulness,
                    "retrieval_precision": retrieval_precision,
                    "retrieval_recall_at_k": retrieval_recall_at_k,
                    "ece": explanation.uncertainty_report.ece or 0.0,
                    "reflection_confidence_gain": reflection_gain,
                    "hallucination_reduction": hallucination_reduction,
                }
            )
            hallucination_scores.append(hallucination_score)
            predicted_confidences.append(explanation.confidence)
            observed_outcomes.append(grounding_score)
        summaries = self._aggregate(per_example)
        summaries["hallucination_rate"] = metrics.hallucination_rate(hallucination_scores)
        if predicted_confidences and observed_outcomes:
            estimator = UncertaintyEstimator()
            summaries["ece"] = estimator.estimate_ece(
                np.array(predicted_confidences, dtype=np.float32),
                np.array(observed_outcomes, dtype=np.float32),
                bins=self.config.get("uncertainty", {}).get("calibration_bins", 10),
            )
        failure_records = None
        failure_summary = None
        eval_cfg = self.config.get("evaluation", {})
        if eval_cfg.get("enable_failure_analysis"):
            analyzer = FailureAnalyzer(thresholds=eval_cfg.get("failure_thresholds"))
            records = analyzer.analyze(explanations, metadata=metadata)
            failure_records = [record.__dict__ for record in records]
            failure_summary = self._summarize_failures(records)
            output_dir = Path(eval_cfg.get("failure_output_dir", "data/failure_analysis"))
            analyzer.export(records, output_dir)
        self._log_wandb(summaries, failure_summary)
        return EvaluationResult(
            summaries=summaries,
            per_example=per_example,
            failure_records=failure_records,
            failure_summary=failure_summary,
        )

    def _aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        keys = results[0].keys()
        summary: Dict[str, float] = {}
        for key in keys:
            summary[key] = sum(item[key] for item in results) / len(results)
        return summary

    @staticmethod
    def _summarize_failures(records: List[object]) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        if not records:
            return summary
        total = len(records)
        for record in records:
            failure_type = getattr(record, "predicted_failure_type", "none")
            summary[f"failure_{failure_type}"] = summary.get(f"failure_{failure_type}", 0.0) + 1.0
        for key in list(summary.keys()):
            summary[key] = summary[key] / total
        return summary

    def _log_wandb(self, summaries: Dict[str, float], failure_summary: Optional[Dict[str, float]]) -> None:
        eval_cfg = self.config.get("evaluation", {})
        wandb_cfg = eval_cfg.get("wandb", {})
        if not wandb_cfg.get("enabled"):
            return
        try:
            import wandb
        except ImportError:
            return
        run = wandb.init(
            project=wandb_cfg.get("project", "multimodal_rag"),
            name=wandb_cfg.get("run_name"),
            reinit=True,
        )
        payload = dict(summaries)
        if failure_summary:
            payload.update(failure_summary)
        wandb.log(payload)
        run.finish()
