"""Multi-hop reasoning over semantic graphs."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.schemas import EvidenceChunk, ReasoningStep


class MultiHopReasoner:
    def __init__(self, max_hops: int = 3, path_weights: Dict[str, float] | None = None) -> None:
        self.max_hops = max_hops
        self.path_weights = path_weights or {
            "edge_confidence": 0.25,
            "retrieval_support": 0.2,
            "semantic_similarity": 0.2,
            "graph_coherence": 0.2,
            "node_importance": 0.15,
        }

    def infer(self, graph: nx.MultiDiGraph, evidence: List[EvidenceChunk]) -> List[ReasoningStep]:
        ranked = self.rank_candidate_paths(graph, evidence)
        if not ranked:
            return self.generate_reasoning_chain(graph, evidence)
        top_path, top_score, node_importance = ranked[0]
        return self._steps_from_path(top_path, evidence, top_score, node_importance)

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

    def score_reasoning_path_features(
        self,
        graph: nx.MultiDiGraph,
        path: List[str],
        evidence: List[EvidenceChunk],
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        if len(path) < 2:
            return 0.0, {}, {}
        edge_confidence = self._edge_confidence(graph, path)
        retrieval_support = self._retrieval_support(graph, path)
        semantic_similarity = self._semantic_similarity(graph, path, evidence)
        graph_coherence = self._graph_coherence(path)
        node_importance = self._node_importance(graph, path)
        feature_scores = {
            "edge_confidence": edge_confidence,
            "retrieval_support": retrieval_support,
            "semantic_similarity": semantic_similarity,
            "graph_coherence": graph_coherence,
            "node_importance": node_importance,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for feature, score in feature_scores.items():
            weight = float(self.path_weights.get(feature, 0.0))
            total_weight += weight
            weighted_sum += weight * score
        if total_weight <= 0.0:
            path_score = 0.0
        else:
            path_score = float(max(0.0, min(1.0, weighted_sum / total_weight)))
        node_importance_scores = {
            node: self._node_importance(graph, [node]) for node in path
        }
        return path_score, feature_scores, node_importance_scores

    def rank_candidate_paths(
        self,
        graph: nx.MultiDiGraph,
        evidence: List[EvidenceChunk],
    ) -> List[Tuple[List[str], float, Dict[str, float]]]:
        candidates = self._candidate_paths(graph)
        ranked: List[Tuple[List[str], float, Dict[str, float]]] = []
        for path in candidates:
            score, _, node_importance = self.score_reasoning_path_features(graph, path, evidence)
            ranked.append((path, score, node_importance))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _steps_from_path(
        self,
        path: List[str],
        evidence: List[EvidenceChunk],
        path_score: float,
        node_importance: Dict[str, float],
    ) -> List[ReasoningStep]:
        steps: List[ReasoningStep] = []
        evidence_ids = [chunk.chunk_id for chunk in evidence]
        for idx, node_id in enumerate(path[: self.max_hops]):
            importance = node_importance.get(node_id, 0.5)
            confidence = float(min(1.0, max(0.0, 0.5 * path_score + 0.5 * importance)))
            steps.append(
                ReasoningStep(
                    step_id=f"step_{idx}",
                    statement=f"Reason about node {node_id}",
                    evidence_ids=evidence_ids,
                    confidence=confidence,
                    dependencies=path[:idx],
                )
            )
        return steps

    def _candidate_paths(self, graph: nx.MultiDiGraph) -> List[List[str]]:
        if graph.number_of_nodes() == 0:
            return []
        if "query" in graph:
            targets = [node for node in graph.nodes if node != "query"]
            paths: List[List[str]] = []
            for target in targets:
                for path in nx.all_simple_paths(graph, source="query", target=target, cutoff=self.max_hops):
                    if len(path) > 1:
                        paths.append(path)
            return paths
        node_ids = list(graph.nodes)
        return [node_ids[: self.max_hops]] if node_ids else []

    def _edge_confidence(self, graph: nx.MultiDiGraph, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0
        weights = []
        for source, target in zip(path[:-1], path[1:]):
            edge_data = graph.get_edge_data(source, target, default={})
            if not edge_data:
                weights.append(0.0)
                continue
            best_weight = max(float(data.get("weight", 0.0)) for data in edge_data.values())
            weights.append(best_weight)
        return float(np.mean(weights)) if weights else 0.0

    def _retrieval_support(self, graph: nx.MultiDiGraph, path: List[str]) -> float:
        if not path:
            return 0.0
        scores = []
        for node in path:
            score = float(graph.nodes[node].get("score", 0.0))
            scores.append(max(0.0, min(1.0, score)))
        return float(np.mean(scores)) if scores else 0.0

    def _semantic_similarity(
        self, graph: nx.MultiDiGraph, path: List[str], evidence: List[EvidenceChunk]
    ) -> float:
        if not path or not evidence:
            return 0.0
        node_texts = []
        for node in path:
            attrs = graph.nodes[node]
            node_texts.append(str(attrs.get("label") or attrs.get("text") or node))
        evidence_blob = " ".join(chunk.text for chunk in evidence)
        corpus = [evidence_blob] + node_texts
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        evidence_vec = matrix[0:1]
        node_vecs = matrix[1:]
        sims = cosine_similarity(node_vecs, evidence_vec).reshape(-1)
        if sims.size == 0:
            return 0.0
        return float(max(0.0, min(1.0, float(np.mean(sims)))))

    def _graph_coherence(self, path: List[str]) -> float:
        if len(path) <= 1:
            return 0.0
        return float(max(0.0, 1.0 - (len(path) - 1) / max(1.0, float(self.max_hops))))

    def _node_importance(self, graph: nx.MultiDiGraph, path: Iterable[str]) -> float:
        if graph.number_of_nodes() == 0:
            return 0.0
        centrality = nx.degree_centrality(graph)
        values = [float(centrality.get(node, 0.0)) for node in path]
        return float(np.mean(values)) if values else 0.0
