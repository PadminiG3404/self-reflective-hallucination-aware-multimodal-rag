"""Experiment runner for paper-ready results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import json
import platform
import time

import numpy as np
from PIL import Image
from io import BytesIO
import torch

from src.data.ingestion import load_dataset
from src.data.webqa import build_webqa_relevance_map, load_webqa_questions, WebQAImageStore
from src.evaluation import metrics
from src.evaluation.failure_analysis import FailureAnalyzer
from src.evaluation.runner import EvaluationRunner
from src.evaluation.visualization import FigureBuilder
from src.utils.schemas import FinalExplanation
from src.utils.seed import set_seed


@dataclass
class ProfileResult:
    name: str
    summaries: Dict[str, float]


class ExperimentRunner:
    def __init__(self, config: dict, rag_factory) -> None:
        self.config = config
        self.rag_factory = rag_factory
        self.repo_root = Path(__file__).resolve().parents[2]
        self.results_dir = Path(self.config.get("results", {}).get("output_dir", "results"))
        self.figures_dir = Path(self.config.get("results", {}).get("figures_dir", "results/figures"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> None:
        set_seed(self.config.get("project", {}).get("seed"))
        setup = self._collect_experiment_setup()
        self._export_setup(setup)
        profile_results, per_profile_examples = self._run_benchmarks()
        self._export_benchmark_tables(profile_results)
        self._export_hallucination_breakdown(per_profile_examples)
        self._export_reflection_impact(per_profile_examples)
        self._export_refinement_impact(per_profile_examples)
        self._export_ablation_table(profile_results)
        self._export_case_studies(per_profile_examples)
        self._export_failure_analysis(per_profile_examples)
        FigureBuilder(self.figures_dir).build_all(profile_results, per_profile_examples)
        self._export_final_report(profile_results, per_profile_examples)

    def _collect_experiment_setup(self) -> Dict[str, object]:
        dataset_cfg = self.config.get("dataset", {})
        setup = {
            "dataset": dataset_cfg.get("path"),
            "format": dataset_cfg.get("format"),
            "split": dataset_cfg.get("split"),
            "max_items": dataset_cfg.get("max_items"),
            "seed": self.config.get("project", {}).get("seed"),
            "device": self.config.get("project", {}).get("device"),
            "config": self.config,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        if dataset_cfg.get("format") == "webqa":
            dataset_path = self._resolve_path(dataset_cfg.get("path"))
            questions = load_webqa_questions(
                dataset_path,
                split=dataset_cfg.get("split"),
                max_items=dataset_cfg.get("max_items"),
            )
            setup["sample_count"] = len(questions)
            setup["image_count"] = len({item.get("image_id") for item in questions if item.get("image_id")})
        else:
            dataset_path = self._resolve_path(dataset_cfg.get("path"))
            items = load_dataset(dataset_path, fmt=dataset_cfg.get("format"), max_items=dataset_cfg.get("max_items"))
            setup["sample_count"] = len(items)
            setup["image_count"] = len({item.image_path for item in items if item.image_path})
        return setup

    def _export_setup(self, setup: Dict[str, object]) -> None:
        json_path = self.results_dir / "experiment_setup.json"
        csv_path = self.results_dir / "experiment_setup.csv"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(setup, handle, indent=2)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for key, value in setup.items():
                writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])

    def _run_benchmarks(self) -> Tuple[List[ProfileResult], Dict[str, dict]]:
        profiles = self.config.get("experiments", {}).get("profiles", {})
        profile_results: List[ProfileResult] = []
        per_profile_examples: Dict[str, dict] = {}
        for name, override in profiles.items():
            profile_config = self._merge_config(self.config, override)
            rag = self.rag_factory(profile_config)
            explanations, start_times, end_times, metadata = self._run_dataset(rag, profile_config)
            runner = EvaluationRunner(config=profile_config)
            eval_result = runner.evaluate(explanations, start_times, end_times, metadata=metadata)
            additional = self._compute_answer_metrics(explanations, metadata)
            summaries = dict(eval_result.summaries)
            summaries.update(additional)
            profile_results.append(ProfileResult(name=name, summaries=summaries))
            per_profile_examples[name] = {
                "explanations": explanations,
                "metadata": metadata,
                "summaries": summaries,
                "per_example": eval_result.per_example,
                "failure_records": eval_result.failure_records,
                "failure_summary": eval_result.failure_summary,
            }
        return profile_results, per_profile_examples

    def _run_dataset(self, rag, config: dict):
        dataset_cfg = config.get("dataset", {})
        explanations: List[FinalExplanation] = []
        start_times: List[float] = []
        end_times: List[float] = []
        metadata: List[Dict[str, object]] = []
        if dataset_cfg.get("format") == "webqa":
            dataset_path = self._resolve_path(dataset_cfg.get("path"))
            image_store = None
            tsv_value = dataset_cfg.get("image_tsv")
            if tsv_value:
                tsv_path = self._resolve_path(tsv_value)
                if tsv_path.exists():
                    image_store = WebQAImageStore(tsv_path)
            questions = load_webqa_questions(
                dataset_path,
                split=dataset_cfg.get("split"),
                max_items=dataset_cfg.get("max_items"),
            )
            relevance_map = build_webqa_relevance_map(
                dataset_path,
                split=dataset_cfg.get("split"),
                max_items=dataset_cfg.get("max_items"),
            )
            for item in questions:
                image_obj = None
                image_id = item.get("image_id")
                if image_id and image_store:
                    image_bytes = image_store.get_image_bytes(str(image_id))
                    if image_bytes is not None:
                        image_obj = Image.open(BytesIO(image_bytes)).convert("RGB")
                start_time = time.time()
                explanation = rag.run(image=image_obj, query=item.get("question"))
                end_time = time.time()
                explanations.append(explanation)
                start_times.append(start_time)
                end_times.append(end_time)
                metadata.append(
                    {
                        "query": item.get("question"),
                        "image_id": image_id,
                        "guid": item.get("guid"),
                        "answers": item.get("answers"),
                        "relevant_chunk_ids": relevance_map.get(item.get("guid"), []),
                    }
                )
        else:
            dataset_path = self._resolve_path(dataset_cfg.get("path"))
            items = load_dataset(dataset_path, fmt=dataset_cfg.get("format"), max_items=dataset_cfg.get("max_items"))
            for item in items:
                image_obj = None
                if item.image_path:
                    image_path = self._resolve_path(item.image_path)
                    if image_path.exists():
                        image_obj = Image.open(image_path).convert("RGB")
                start_time = time.time()
                explanation = rag.run(image=image_obj, query=item.text)
                end_time = time.time()
                explanations.append(explanation)
                start_times.append(start_time)
                end_times.append(end_time)
                metadata.append({"query": item.text, "answers": [item.text]})
        return explanations, start_times, end_times, metadata

    def _compute_answer_metrics(self, explanations: List[FinalExplanation], metadata: List[Dict[str, object]]) -> Dict[str, float]:
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        mrrs = []
        contradiction_hits = []
        contradiction_threshold = self.config.get("hallucination", {}).get("contradiction_threshold", 0.5)
        for explanation, meta in zip(explanations, metadata):
            answers = meta.get("answers") or []
            if not isinstance(answers, list):
                answers = [answers]
            predicted = explanation.answer or ""
            accuracies.append(metrics.exact_match(predicted, answers))
            precision, recall, f1 = metrics.token_f1(predicted, answers)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            relevant_ids = meta.get("relevant_chunk_ids") or []
            if relevant_ids:
                retrieved_ids = [chunk.chunk_id for chunk in explanation.evidence_chain]
                mrrs.append(metrics.mean_reciprocal_rank(retrieved_ids, relevant_ids))
            contradiction_score = explanation.hallucination_report.factor_scores.get("contradiction", 0.0)
            contradiction_hits.append(1.0 if contradiction_score >= contradiction_threshold else 0.0)
        return {
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "contradiction_detection_rate": float(np.mean(contradiction_hits)) if contradiction_hits else 0.0,
        }

    def _export_benchmark_tables(self, profile_results: List[ProfileResult]) -> None:
        csv_path = self.results_dir / "benchmark_table.csv"
        json_path = self.results_dir / "benchmark_summary.json"
        rows = []
        for result in profile_results:
            row = {"model": result.name}
            row.update(result.summaries)
            rows.append(row)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary = {result.name: result.summaries for result in profile_results}
        visage = summary.get("visage_x")
        full = summary.get("full_framework")
        if visage and full:
            summary["delta_over_visage_x"] = {
                key: full.get(key, 0.0) - visage.get(key, 0.0) for key in full.keys()
            }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def _export_hallucination_breakdown(self, per_profile_examples: Dict[str, dict]) -> None:
        csv_path = self.results_dir / "hallucination_breakdown.csv"
        rows = []
        for name, payload in per_profile_examples.items():
            explanations = payload["explanations"]
            breakdown = self._hallucination_categories(explanations)
            for category, stats in breakdown.items():
                row = {"model": name, "category": category}
                row.update(stats)
                rows.append(row)
        if not rows:
            return
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _hallucination_categories(self, explanations: List[FinalExplanation]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        category_examples: Dict[str, List[FinalExplanation]] = {}
        for explanation in explanations:
            report = explanation.hallucination_report
            category = self._map_hallucination_category(report)
            counts[category] = counts.get(category, 0) + 1
            stats.setdefault(category, []).append(report.hallucination_score)
            category_examples.setdefault(category, []).append(explanation)
        result: Dict[str, Dict[str, float]] = {}
        total = max(1, len(explanations))
        for category, scores in stats.items():
            examples = category_examples.get(category, [])
            result[category] = {
                "frequency": counts.get(category, 0) / total,
                "avg_severity": float(np.mean(scores)) if scores else 0.0,
                "avg_uncertainty": float(
                    np.mean([ex.uncertainty_report.calibration_score for ex in examples])
                )
                if examples
                else 0.0,
                "avg_grounding": float(
                    np.mean([1.0 - ex.hallucination_report.hallucination_score for ex in examples])
                )
                if examples
                else 0.0,
            }
        return result

    def _export_reflection_impact(self, per_profile_examples: Dict[str, dict]) -> None:
        csv_path = self.results_dir / "reflection_impact.csv"
        rows = []
        for name, payload in per_profile_examples.items():
            for explanation in payload["explanations"]:
                report = explanation.reflection_report
                if report is None:
                    continue
                rows.append(
                    {
                        "model": name,
                        "hallucination_score": explanation.hallucination_report.hallucination_score,
                        "confidence": explanation.confidence,
                        "grounding_score": 1.0 - explanation.hallucination_report.hallucination_score,
                        "retrieval_drift": explanation.hallucination_report.factor_scores.get("retrieval_drift", 0.0),
                        "answer_accuracy": explanation.confidence,
                        "ece": explanation.uncertainty_report.ece or 0.0,
                        "hallucination_delta": report.hallucination_reduction,
                        "confidence_gain": report.confidence_improvement,
                        "retrieval_improvement": report.retrieval_improvement,
                    }
                )
        if not rows:
            return
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _export_refinement_impact(self, per_profile_examples: Dict[str, dict]) -> None:
        csv_path = self.results_dir / "refinement_impact.csv"
        rows = []
        for name, payload in per_profile_examples.items():
            for explanation in payload["explanations"]:
                report = explanation.reflection_report
                if report is None:
                    continue
                rows.append(
                    {
                        "model": name,
                        "retrieval_precision": payload["summaries"].get("retrieval_precision", 0.0),
                        "retrieval_recall_at_k": payload["summaries"].get("retrieval_recall_at_k", 0.0),
                        "evidence_confidence": explanation.retrieval_metrics.evidence_confidence
                        if explanation.retrieval_metrics
                        else 0.0,
                        "retrieval_drift": explanation.hallucination_report.factor_scores.get("retrieval_drift", 0.0),
                        "hallucination_delta": report.hallucination_reduction,
                    }
                )
        if not rows:
            return
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _export_ablation_table(self, profile_results: List[ProfileResult]) -> None:
        csv_path = self.results_dir / "ablation_table.csv"
        rows = []
        for result in profile_results:
            row = {"model": result.name}
            row.update(result.summaries)
            rows.append(row)
        if not rows:
            return
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _export_case_studies(self, per_profile_examples: Dict[str, dict]) -> None:
        json_path = self.results_dir / "case_studies.json"
        cases = []
        for name, payload in per_profile_examples.items():
            examples = list(zip(payload["explanations"], payload["metadata"]))
            cases.extend(self._select_case_studies(name, examples))
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(cases, handle, indent=2)

    def _export_failure_analysis(self, per_profile_examples: Dict[str, dict]) -> None:
        csv_path = self.results_dir / "failure_analysis.csv"
        rows = []
        for name, payload in per_profile_examples.items():
            explanations = payload["explanations"]
            metadata = payload["metadata"]
            analyzer = FailureAnalyzer(thresholds=self.config.get("evaluation", {}).get("failure_thresholds"))
            records = analyzer.analyze(explanations, metadata=metadata)
            summary = self._summarize_failure_records(records, payload.get("per_example", []))
            for failure_type, stats in summary.items():
                rows.append(
                    {
                        "model": name,
                        "failure_type": failure_type,
                        "frequency": stats["frequency"],
                        "avg_hallucination": stats["avg_hallucination"],
                        "avg_uncertainty": stats["avg_uncertainty"],
                        "avg_latency": stats["avg_latency"],
                        "example_ids": ";".join(stats["example_ids"]),
                    }
                )
        if not rows:
            return
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _export_final_report(self, profile_results: List[ProfileResult], per_profile_examples: Dict[str, dict]) -> None:
        report_path = self.results_dir / "final_results_report.md"
        best = max(profile_results, key=lambda item: item.summaries.get("f1", 0.0), default=None)
        visage = next((item for item in profile_results if item.name == "visage_x"), None)
        full = next((item for item in profile_results if item.name == "full_framework"), None)
        lines = ["# Final Results Report", ""]
        if best:
            lines.append(f"- best_model: {best.name}")
        if visage and full:
            delta = full.summaries.get("hallucination_rate", 0.0) - visage.summaries.get("hallucination_rate", 0.0)
            lines.append(f"- delta_vs_visage_x_hallucination_rate: {delta:.4f}")
        if profile_results:
            worst = max(profile_results, key=lambda item: item.summaries.get("hallucination_rate", 0.0))
            best_h = min(profile_results, key=lambda item: item.summaries.get("hallucination_rate", 0.0))
            reduction = worst.summaries.get("hallucination_rate", 0.0) - best_h.summaries.get("hallucination_rate", 0.0)
            lines.append(f"- largest_hallucination_reduction: {reduction:.4f}")
        if full:
            impacts = []
            for item in profile_results:
                if item.name == full.name:
                    continue
                impacts.append((item.name, full.summaries.get("f1", 0.0) - item.summaries.get("f1", 0.0)))
            impacts.sort(key=lambda pair: pair[1], reverse=True)
            if impacts:
                lines.append(f"- most_impactful_component: {impacts[0][0]} (delta_f1={impacts[0][1]:.4f})")
        if full:
            failure_summary = per_profile_examples.get(full.name, {}).get("failure_summary") or {}
            if failure_summary:
                biggest = max(failure_summary.items(), key=lambda kv: kv[1])
                lines.append(f"- biggest_failure_mode: {biggest[0]}")
        if full:
            per_example = per_profile_examples.get(full.name, {}).get("per_example", [])
            if per_example:
                reflection_gain = float(np.mean([item["reflection_confidence_gain"] for item in per_example]))
                lines.append(f"- reflection_gain: {reflection_gain:.4f}")
        if visage and full:
            delta_ece = full.summaries.get("ece", 0.0) - visage.summaries.get("ece", 0.0)
            lines.append(f"- calibration_improvements: {delta_ece:.4f}")
        if visage and full:
            tradeoffs = []
            for key in ["latency_ms", "retrieval_precision", "retrieval_recall_at_k", "f1"]:
                if full.summaries.get(key, 0.0) < visage.summaries.get(key, 0.0):
                    tradeoffs.append(key)
            lines.append(f"- tradeoffs: {', '.join(tradeoffs) if tradeoffs else 'none'}")
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _map_hallucination_category(self, report) -> str:
        dominant = report.dominant_factor or report.hallucination_type
        if dominant == "retrieval_drift":
            return "retrieval_drift"
        if dominant == "contradiction":
            return "contradiction_miss"
        if dominant == "graph_inconsistency":
            return "cross_modal_conflict"
        if dominant == "grounding":
            return "unsupported_claim"
        return "semantic_fabrication"

    def _resolve_path(self, path_value: str | Path | None) -> Path:
        if not path_value:
            return Path("")
        resolved = Path(path_value)
        if resolved.is_absolute():
            return resolved
        return self.repo_root / resolved

    def _select_case_studies(self, model_name: str, examples: List[Tuple[FinalExplanation, Dict[str, object]]]) -> List[Dict[str, object]]:
        cases = []
        def build_case(label: str, explanation: FinalExplanation, meta: Dict[str, object]) -> Dict[str, object]:
            return {
                "model": model_name,
                "label": label,
                "query": meta.get("query"),
                "image_id": meta.get("image_id"),
                "retrieved_chunks": [chunk.text for chunk in explanation.evidence_chain[:5]],
                "reasoning_chain": [step.statement for step in explanation.reasoning_steps],
                "reflection": explanation.reflection_report.model_dump()
                if explanation.reflection_report
                else None,
                "final_answer": explanation.answer,
                "hallucination_score": explanation.hallucination_report.hallucination_score,
                "uncertainty_score": explanation.uncertainty_report.calibration_score,
                "trace_summary": explanation.trace_summary,
            }
        corrected = [ex for ex in examples if ex[0].reflection_report and ex[0].reflection_report.hallucination_reduction > 0]
        if corrected:
            cases.append(build_case("corrected_hallucination", corrected[0][0], corrected[0][1]))
        drift = [ex for ex in examples if ex[0].hallucination_report.factor_scores.get("retrieval_drift", 0.0) > 0.3]
        if drift:
            cases.append(build_case("retrieval_drift_recovery", drift[0][0], drift[0][1]))
        graph = [ex for ex in examples if ex[0].trace_summary]
        if graph:
            cases.append(build_case("graph_reasoning_trace", graph[0][0], graph[0][1]))
        confidence = [ex for ex in examples if ex[0].reflection_report and ex[0].reflection_report.confidence_improvement > 0]
        if confidence:
            cases.append(build_case("confidence_correction", confidence[0][0], confidence[0][1]))
        failures = [ex for ex in examples if ex[0].hallucination_report.hallucination_score > 0.6]
        if failures:
            cases.append(build_case("failure_example", failures[0][0], failures[0][1]))
        return cases

    @staticmethod
    def _summarize_failure_records(records, per_example: List[Dict[str, float]]) -> Dict[str, Dict[str, object]]:
        summary: Dict[str, Dict[str, object]] = {}
        counts: Dict[str, int] = {}
        for idx, record in enumerate(records):
            failure_type = record.predicted_failure_type
            counts[failure_type] = counts.get(failure_type, 0) + 1
            summary.setdefault(failure_type, {
                "hallucination_scores": [],
                "uncertainty_scores": [],
                "latencies": [],
                "example_ids": [],
            })
            summary[failure_type]["hallucination_scores"].append(record.hallucination_score)
            summary[failure_type]["uncertainty_scores"].append(record.uncertainty_score)
            latency_value = per_example[idx].get("latency_ms", 0.0) if idx < len(per_example) else 0.0
            summary[failure_type]["latencies"].append(latency_value)
            summary[failure_type]["example_ids"].append(record.image_id or record.query[:32] or "unknown")
        total = max(1, len(records))
        result: Dict[str, Dict[str, object]] = {}
        for failure_type, stats in summary.items():
            result[failure_type] = {
                "frequency": counts.get(failure_type, 0) / total,
                "avg_hallucination": float(np.mean(stats["hallucination_scores"])) if stats["hallucination_scores"] else 0.0,
                "avg_uncertainty": float(np.mean(stats["uncertainty_scores"])) if stats["uncertainty_scores"] else 0.0,
                "avg_latency": float(np.mean(stats["latencies"])) if stats["latencies"] else 0.0,
                "example_ids": stats["example_ids"][:3],
            }
        return result

    @staticmethod
    def _merge_config(base: dict, override: dict) -> dict:
        merged = json.loads(json.dumps(base))
        stack = [(merged, override)]
        while stack:
            target, source = stack.pop()
            for key, value in source.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    stack.append((target[key], value))
                else:
                    target[key] = value
        return merged
