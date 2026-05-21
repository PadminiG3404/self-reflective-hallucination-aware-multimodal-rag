"""Lightweight entity extraction for graph enrichment."""
from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


class EntityExtractor:
    def __init__(self, top_k: int = 3, min_len: int = 3) -> None:
        self.top_k = top_k
        self.min_len = min_len

    def extract_entities(self, text: str) -> List[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([cleaned])
        scores = tfidf.toarray().reshape(-1)
        tokens = vectorizer.get_feature_names_out()
        ranked = sorted(zip(tokens, scores), key=lambda item: item[1], reverse=True)
        entities = [token for token, _ in ranked if len(token) >= self.min_len]
        return entities[: self.top_k]
