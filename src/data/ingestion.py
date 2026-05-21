"""Dataset ingestion and evidence chunk building."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.data.chunking import chunk_text
from src.data.schema import DatasetItem
from src.utils.schemas import EvidenceChunk


def load_dataset(path: Path, fmt: Optional[str] = None, max_items: Optional[int] = None) -> List[DatasetItem]:
    if not path.exists():
        return []
    format_name = fmt or path.suffix.lstrip(".").lower()
    if format_name == "jsonl":
        items = _load_jsonl(path)
    elif format_name == "json":
        items = _load_json(path)
    else:
        raise ValueError(f"Unsupported dataset format: {format_name}")
    if max_items is not None:
        items = items[:max_items]
    return items


def build_evidence_chunks(
    items: List[DatasetItem],
    chunk_size: int,
    overlap: int,
    source: str,
) -> List[EvidenceChunk]:
    evidence: List[EvidenceChunk] = []
    for item in items:
        chunks = chunk_text(item.text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{item.item_id}_chunk_{idx}"
            evidence.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    text=chunk,
                    source=source,
                    score=0.0,
                    metadata={
                        "item_id": item.item_id,
                        "image_path": item.image_path,
                        "chunk_index": idx,
                        **item.metadata,
                    },
                )
            )
    return evidence


def _load_json(path: Path) -> List[DatasetItem]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("JSON dataset must be a list of items")
    return [_parse_item(item, idx) for idx, item in enumerate(raw)]


def _load_jsonl(path: Path) -> List[DatasetItem]:
    items: List[DatasetItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            items.append(_parse_item(json.loads(line), idx))
    return items


def _parse_item(raw: Dict[str, Any], idx: int) -> DatasetItem:
    item_id = str(raw.get("item_id") or raw.get("id") or f"item_{idx}")
    text = str(raw.get("text") or raw.get("caption") or raw.get("content") or "")
    image_path = raw.get("image_path") or raw.get("image")
    metadata = raw.get("metadata") or {}
    return DatasetItem(item_id=item_id, text=text, image_path=image_path, metadata=metadata)
