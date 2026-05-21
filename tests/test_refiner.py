import numpy as np

from src.graph_reasoning.semantic_graph import SemanticReasoningGraph
from src.refinement.retrieval_refiner import AdaptiveRetrieverRefiner
from src.retrieval.faiss_retriever import CrossModalRetriever
from src.utils.schemas import EvidenceChunk


def test_refiner_updates_graph():
    retriever = CrossModalRetriever(embedding_dim=2)
    embeddings = np.random.randn(3, 2).astype(np.float32)
    evidence = [
        EvidenceChunk(chunk_id=str(i), text=f"chunk {i}", source="unit", score=0.0)
        for i in range(3)
    ]
    retriever.build_index(embeddings, evidence)
    refiner = AdaptiveRetrieverRefiner(retriever)
    graph = SemanticReasoningGraph()
    query = np.random.randn(1, 2).astype(np.float32)
    refiner.refine(query, top_k=2, graph=graph)
    assert len(graph.graph.nodes) == 2
