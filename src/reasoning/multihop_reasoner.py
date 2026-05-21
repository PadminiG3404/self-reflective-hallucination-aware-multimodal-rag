"""Multi-hop reasoning over semantic graphs."""
from __future__ import annotations

from typing import Dict, List

import networkx as nx

from src.utils.schemas import EvidenceChunk, ReasoningStep


class MultiHopReasoner:
    def __init__(self, max_hops: int = 3) -> None:
        self.max_hops = max_hops

    def infer(self, graph: nx.MultiDiGraph, evidence: List[EvidenceChunk]) -> List[ReasoningStep]:
        reasoning_chain = self.generate_reasoning_chain(graph, evidence)
        return reasoning_chain

    def generate_reasoning_chain(
        self, graph: nx.MultiDiGraph, evidence: List[EvidenceChunk]
    ) -> List[ReasoningStep]:
        steps: List[ReasoningStep] = []
        node_ids = list(graph.nodes)
        for idx, node_id in enumerate(node_ids[: self.max_hops]):
            steps.append(
                ReasoningStep(
                    step_id=f"step_{idx}",
                    statement=f"Reason about node {node_id}",
                    evidence_ids=[chunk.chunk_id for chunk in evidence],
                    confidence=0.5,
                    dependencies=node_ids[:idx],
                )
            )
        return steps

    def score_reasoning_path(self, steps: List[ReasoningStep]) -> float:
        if not steps:
            return 0.0
        return float(sum(step.confidence for step in steps) / len(steps))
