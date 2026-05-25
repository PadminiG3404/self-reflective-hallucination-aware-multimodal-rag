import numpy as np

from src.graph_reasoning.semantic_graph import SemanticReasoningGraph
from src.refinement.retrieval_refiner import AdaptiveRetrieverRefiner
from src.retrieval.faiss_retriever import CrossModalRetriever
from src.utils.schemas import EvidenceChunk, ReflectionReport


class _StubEncoder:
    def encode_text(self, texts):
        array = np.random.randn(len(texts), 2).astype(np.float32)

        class _Embeddings:
            def __init__(self, values):
                self._values = values

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        return type("Enc", (), {"embeddings": _Embeddings(array)})


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


def test_structured_refine_returns_query():
    retriever = CrossModalRetriever(embedding_dim=2)
    embeddings = np.random.randn(3, 2).astype(np.float32)
    evidence = [
        EvidenceChunk(chunk_id=str(i), text=f"chunk {i}", source="unit", score=0.1)
        for i in range(3)
    ]
    retriever.build_index(embeddings, evidence)
    refiner = AdaptiveRetrieverRefiner(retriever, encoder=_StubEncoder())
    graph = SemanticReasoningGraph()
    report = ReflectionReport(invalid_steps=["s1"], missing_evidence=["s2"])
    result, revised_query, improvement = refiner.structured_refine(
        "query", evidence, top_k=2, graph=graph, reflection_report=report
    )
    assert result.chunks
    assert "Focus" in revised_query
    assert improvement >= 0.0
