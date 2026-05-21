"""Adaptive retrieval refinement loop."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.encoders.multimodal_encoder import MultimodalEncoder
from src.graph_reasoning.semantic_graph import GraphNode, SemanticReasoningGraph
from src.retrieval.faiss_retriever import CrossModalRetriever, RetrievalResult
from src.utils.schemas import EvidenceChunk


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
