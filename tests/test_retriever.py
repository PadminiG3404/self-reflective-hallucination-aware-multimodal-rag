import numpy as np

from src.retrieval.faiss_retriever import CrossModalRetriever
from src.utils.schemas import EvidenceChunk


def test_retriever_returns_top_k():
    retriever = CrossModalRetriever(embedding_dim=4)
    embeddings = np.random.randn(5, 4).astype(np.float32)
    evidence = [
        EvidenceChunk(chunk_id=str(i), text=f"chunk {i}", source="unit", score=0.0)
        for i in range(5)
    ]
    retriever.build_index(embeddings, evidence)
    query = np.random.randn(1, 4).astype(np.float32)
    result = retriever.retrieve(query, top_k=3)
    assert len(result.chunks) == 3
    assert result.metrics.evidence_confidence >= result.metrics.alignment_score
