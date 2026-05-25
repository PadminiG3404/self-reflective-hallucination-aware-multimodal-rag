"""Matplotlib-based figure generation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


class FigureBuilder:
    def __init__(self, figures_dir: Path) -> None:
        self.figures_dir = figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def build_all(self, profile_results: List[object], per_profile_examples: Dict[str, dict]) -> None:
        self._benchmark_chart(profile_results)
        self._ablation_chart(profile_results)
        self._hallucination_breakdown(per_profile_examples)
        self._reflection_gain_chart(per_profile_examples)
        self._calibration_curve(per_profile_examples)
        self._failure_distribution(per_profile_examples)

    def _benchmark_chart(self, profile_results: List[object]) -> None:
        names = [result.name for result in profile_results]
        scores = [result.summaries.get("f1", 0.0) for result in profile_results]
        plt.figure(figsize=(8, 4))
        plt.bar(names, scores)
        plt.ylabel("F1")
        plt.title("Benchmark Comparison")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "benchmark_comparison.png")
        plt.close()

    def _ablation_chart(self, profile_results: List[object]) -> None:
        names = [result.name for result in profile_results]
        scores = [result.summaries.get("hallucination_rate", 0.0) for result in profile_results]
        plt.figure(figsize=(8, 4))
        plt.bar(names, scores)
        plt.ylabel("Hallucination Rate")
        plt.title("Ablation Contribution")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "ablation_contribution.png")
        plt.close()

    def _hallucination_breakdown(self, per_profile_examples: Dict[str, dict]) -> None:
        if "full_framework" not in per_profile_examples:
            return
        explanations = per_profile_examples["full_framework"]["explanations"]
        counts: Dict[str, int] = {}
        for explanation in explanations:
            report = explanation.hallucination_report
            dominant = report.dominant_factor or report.hallucination_type
            counts[dominant] = counts.get(dominant, 0) + 1
        if not counts:
            return
        labels = list(counts.keys())
        values = [counts[label] for label in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.ylabel("Count")
        plt.title("Hallucination Breakdown (Full Framework)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "hallucination_breakdown.png")
        plt.close()

    def _reflection_gain_chart(self, per_profile_examples: Dict[str, dict]) -> None:
        names = []
        gains = []
        for name, payload in per_profile_examples.items():
            per_example = payload.get("per_example", [])
            if not per_example:
                continue
            names.append(name)
            gains.append(sum(item["reflection_confidence_gain"] for item in per_example) / len(per_example))
        if not names:
            return
        plt.figure(figsize=(8, 4))
        plt.bar(names, gains)
        plt.ylabel("Avg Confidence Gain")
        plt.title("Reflection Impact")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "reflection_impact.png")
        plt.close()

    def _calibration_curve(self, per_profile_examples: Dict[str, dict]) -> None:
        if "full_framework" not in per_profile_examples:
            return
        per_example = per_profile_examples["full_framework"].get("per_example", [])
        if not per_example:
            return
        uncertainties = [item["uncertainty"] for item in per_example]
        groundings = [item["grounding_score"] for item in per_example]
        bins = 10
        bin_edges = [i / bins for i in range(bins + 1)]
        bin_means = []
        bin_obs = []
        for i in range(bins):
            lower, upper = bin_edges[i], bin_edges[i + 1]
            idxs = [j for j, u in enumerate(uncertainties) if lower <= u < upper]
            if not idxs:
                continue
            bin_means.append(sum(uncertainties[j] for j in idxs) / len(idxs))
            bin_obs.append(sum(groundings[j] for j in idxs) / len(idxs))
        if not bin_means:
            return
        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.plot(bin_means, bin_obs, marker="o")
        plt.xlabel("Predicted Uncertainty")
        plt.ylabel("Observed Grounding")
        plt.title("Calibration Curve")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "calibration_curve.png")
        plt.close()

    def _failure_distribution(self, per_profile_examples: Dict[str, dict]) -> None:
        if "full_framework" not in per_profile_examples:
            return
        failure_summary = per_profile_examples["full_framework"].get("failure_summary") or {}
        if not failure_summary:
            return
        labels = list(failure_summary.keys())
        values = [failure_summary[label] for label in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.ylabel("Frequency")
        plt.title("Failure Distribution (Full Framework)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(self.figures_dir / "failure_distribution.png")
        plt.close()
