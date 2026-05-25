"""Adaptive retrieval refinement loop."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.encoders.multimodal_encoder import MultimodalEncoder
from src.graph_reasoning.semantic_graph import GraphNode, SemanticReasoningGraph
from src.retrieval.faiss_retriever import CrossModalRetriever, RetrievalResult
from src.utils.schemas import EvidenceChunk, ReflectionReport


class AdaptiveRetrieverRefiner:
    def __init__(self, retriever: CrossModalRetriever, encoder: Optional[MultimodalEncoder] = None) -> None:
        self.retriever = retriever
        self.encoder = encoder

    def refine(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        graph: SemanticReasoningGraph,
    ) -> RetrievalResult:
        return self.requery(query_embedding, top_k, graph)

    def refine_with_text(
        self,
        query: str,
        chunks: List[EvidenceChunk],
        top_k: int,
        graph: SemanticReasoningGraph,
    ) -> RetrievalResult:
        if self.encoder is None:
            raise RuntimeError("Encoder is required for text-based refinement")
        refined_query = self.reformulate_query(query, chunks)
        query_embedding = self.encoder.encode_text([refined_query]).embeddings.cpu().numpy()
        return self.requery(query_embedding, top_k, graph)

    def structured_refine(
        self,
        query: str,
        chunks: List[EvidenceChunk],
        top_k: int,
        graph: SemanticReasoningGraph,
        reflection_report: Optional[ReflectionReport] = None,
    ) -> Tuple[RetrievalResult, str, float]:
        if self.encoder is None:
            raise RuntimeError("Encoder is required for text-based refinement")
        revised_query = self.build_revised_query(query, chunks, reflection_report)
        query_embedding = self.encoder.encode_text([revised_query]).embeddings.cpu().numpy()
        before_confidence = max((chunk.score for chunk in chunks), default=0.0)
        result = self.requery(query_embedding, top_k, graph)
        after_confidence = max((chunk.score for chunk in result.chunks), default=0.0)
        retrieval_improvement = float(max(0.0, after_confidence - before_confidence))
        return result, revised_query, retrieval_improvement

    def requery(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        graph: SemanticReasoningGraph,
    ) -> RetrievalResult:
        result = self.retriever.retrieve(query_embedding, top_k)
        self.update_graph(graph, result.chunks)
        return result

    def update_graph(self, graph: SemanticReasoningGraph, chunks: List[EvidenceChunk]) -> None:
        for chunk in chunks:
            graph.add_node(
                node=GraphNode(
                    node_id=f"chunk_{chunk.chunk_id}",
                    node_type="retrieved_chunk",
                    attributes={"score": chunk.score},
                )
            )

    @staticmethod
    def reformulate_query(query: str, chunks: List[EvidenceChunk]) -> str:
        if not chunks:
            return query
        top_text = " ".join(chunk.text for chunk in chunks[:2])
        return f"{query} Context: {top_text}"

    @staticmethod
    def build_revised_query(
        query: str,
        chunks: List[EvidenceChunk],
        reflection_report: Optional[ReflectionReport],
    ) -> str:
        if reflection_report is None:
            return AdaptiveRetrieverRefiner.reformulate_query(query, chunks)
        critique_terms = []
        if reflection_report.invalid_steps:
            critique_terms.append("weak reasoning")
        if reflection_report.missing_evidence:
            critique_terms.append("missing evidence")
        focus = " and ".join(critique_terms) if critique_terms else "additional evidence"
        context = " ".join(chunk.text for chunk in chunks[:2]) if chunks else ""
        if context:
            return f"{query} Focus: {focus}. Context: {context}"
        return f"{query} Focus: {focus}."
