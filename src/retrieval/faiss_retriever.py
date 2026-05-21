"""FAISS-based cross-modal retriever."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from src.utils.schemas import EvidenceChunk, RetrievalMetrics


@dataclass
class RetrievalResult:
    chunks: List[EvidenceChunk]
    metrics: RetrievalMetrics


class CrossModalRetriever:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self.index = None
        self.image_index = None
        self._evidence: List[EvidenceChunk] = []
        self._embeddings: np.ndarray | None = None
        self._image_embeddings: np.ndarray | None = None

    def build_index(
        self,
        embeddings: np.ndarray,
        evidence: List[EvidenceChunk],
        image_embeddings: np.ndarray | None = None,
    ) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embeddings have incorrect shape")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))
        self._evidence = evidence
        self._embeddings = embeddings
        if image_embeddings is not None:
            if image_embeddings.shape != embeddings.shape:
                raise ValueError("Image embeddings have incorrect shape")
            self.image_index = faiss.IndexFlatIP(self.embedding_dim)
            self.image_index.add(image_embeddings.astype(np.float32))
            self._image_embeddings = image_embeddings

    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> RetrievalResult:
        if self.index is None or self._embeddings is None:
            raise RuntimeError("Index not built")
        query_embedding = query_embedding.astype(np.float32)
        max_k = min(top_k, len(self._evidence))
        scores, indices = self.index.search(query_embedding, max_k)
        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._evidence[int(idx)]
            if hasattr(chunk, "model_copy"):
                chunks.append(chunk.model_copy(update={"score": float(score)}))
            else:
                chunks.append(chunk.copy(update={"score": float(score)}))
        metrics = self.compute_reliability(scores[0])
        return RetrievalResult(chunks=chunks, metrics=metrics)

    def retrieve_multimodal(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray | None,
        top_k: int,
        alpha: float = 0.5,
    ) -> RetrievalResult:
        if self.index is None or self._embeddings is None:
            raise RuntimeError("Index not built")
        text_embedding = text_embedding.astype(np.float32)
        max_k = min(top_k, len(self._evidence))
        text_scores, text_indices = self.index.search(text_embedding, max_k)
        if image_embedding is None or self.image_index is None:
            return self.retrieve(text_embedding, top_k)
        image_embedding = image_embedding.astype(np.float32)
        image_scores, image_indices = self.image_index.search(image_embedding, max_k)
        fused_scores, fused_indices = self._fuse_scores(
            text_scores[0], text_indices[0], image_scores[0], image_indices[0], alpha
        )
        chunks = []
        for score, idx in zip(fused_scores, fused_indices):
            if idx < 0:
                continue
            chunk = self._evidence[int(idx)]
            if hasattr(chunk, "model_copy"):
                chunks.append(chunk.model_copy(update={"score": float(score)}))
            else:
                chunks.append(chunk.copy(update={"score": float(score)}))
        metrics = self.compute_reliability(np.array(fused_scores, dtype=np.float32))
        return RetrievalResult(chunks=chunks, metrics=metrics)

    def compute_reliability(self, scores: np.ndarray) -> RetrievalMetrics:
        normalized = self._normalize_scores(scores)
        semantic_similarity = float(np.mean(normalized)) if normalized.size else 0.0
        retrieval_consistency = float(np.std(normalized)) if normalized.size else 0.0
        evidence_confidence = float(np.max(normalized)) if normalized.size else 0.0
        alignment_score = float(np.min(normalized)) if normalized.size else 0.0
        return RetrievalMetrics(
            semantic_similarity=semantic_similarity,
            retrieval_consistency=retrieval_consistency,
            evidence_confidence=evidence_confidence,
            alignment_score=alignment_score,
        )

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        normalized = (scores + 1.0) / 2.0
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def _fuse_scores(
        text_scores: np.ndarray,
        text_indices: np.ndarray,
        image_scores: np.ndarray,
        image_indices: np.ndarray,
        alpha: float,
    ) -> tuple[list[float], list[int]]:
        alpha = max(0.0, min(1.0, alpha))
        score_map: dict[int, float] = {}
        for score, idx in zip(text_scores, text_indices):
            if idx < 0:
                continue
            score_map[int(idx)] = alpha * float(score)
        for score, idx in zip(image_scores, image_indices):
            if idx < 0:
                continue
            score_map[int(idx)] = score_map.get(int(idx), 0.0) + (1.0 - alpha) * float(score)
        fused = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        fused_scores = [item[1] for item in fused]
        fused_indices = [item[0] for item in fused]
        return fused_scores, fused_indices
