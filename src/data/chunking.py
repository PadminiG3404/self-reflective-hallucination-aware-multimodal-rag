"""Text chunking utilities."""
from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - max(0, overlap))
    chunks: List[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks
