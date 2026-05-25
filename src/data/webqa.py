"""WebQA dataset ingestion and image access utilities."""
from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image

from src.utils.schemas import EvidenceChunk


class WebQAImageStore:
    def __init__(
        self,
        tsv_path: Path,
        index_path: Optional[Path] = None,
        target_ids: Optional[set[str]] = None,
    ) -> None:
        self.tsv_path = tsv_path
        self.index_path = index_path or tsv_path.with_suffix(".index.json")
        self._index: Dict[str, int] = {}
        self._target_ids = target_ids
        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as handle:
                self._index = json.load(handle)
            return
        if self._target_ids:
            self._index = self._build_partial_index(self._target_ids)
            return
        self._index = {}

    def _build_partial_index(self, target_ids: set[str]) -> Dict[str, int]:
        index: Dict[str, int] = {}
        remaining = set(str(item) for item in target_ids if item)
        if not remaining:
            return index
        with self.tsv_path.open("rb") as handle:
            while remaining:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                parts = line.split(b"\t", 1)
                if not parts:
                    continue
                image_id = parts[0].decode("utf-8", errors="ignore").strip()
                if image_id in remaining:
                    index[image_id] = offset
                    remaining.discard(image_id)
        return index

    def get_image_bytes(self, image_id: str) -> Optional[bytes]:
        image_key = str(image_id)
        offset = self._index.get(image_key)
        if offset is None and not self._target_ids:
            self._index.update(self._build_partial_index({image_key}))
            offset = self._index.get(image_key)
        if offset is None:
            return None
        with self.tsv_path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            return None
        encoded = parts[1].strip()
        try:
            return base64.b64decode(encoded)
        except base64.binascii.Error:
            return None

    def get_image(self, image_id: str) -> Optional[Image.Image]:
        image_bytes = self.get_image_bytes(image_id)
        if image_bytes is None:
            return None
        return Image.open(BytesIO(image_bytes)).convert("RGB")


class WebQALoader:
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def iter_examples(self) -> Iterable[dict]:
        with self.data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for guid, sample in data.items():
            yield guid, sample


def build_webqa_evidence(
    data_path: Path,
    split: Optional[str] = None,
    include_negatives: bool = False,
    max_items: Optional[int] = None,
) -> List[EvidenceChunk]:
    loader = WebQALoader(data_path)
    evidence: List[EvidenceChunk] = []
    count = 0
    for guid, sample in loader.iter_examples():
        if split and sample.get("split") != split:
            continue
        count += 1
        if max_items is not None and count > max_items:
            break
        facts = list(sample.get("img_posFacts", []))
        if include_negatives:
            facts.extend(sample.get("img_negFacts", []))
        for idx, fact in enumerate(facts):
            image_id = str(fact.get("image_id") or "")
            caption = str(fact.get("caption") or "").strip()
            title = str(fact.get("title") or "").strip()
            text = caption or title
            if not text:
                continue
            chunk_id = f"{guid}_img_{image_id}_{idx}"
            evidence.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source="webqa",
                    score=0.0,
                    metadata={
                        "guid": guid,
                        "question": sample.get("Q"),
                        "answer": sample.get("A"),
                        "image_id": image_id,
                        "image_source": "webqa_tsv",
                        "split": sample.get("split"),
                        "topic": sample.get("topic"),
                    },
                )
            )
    return evidence


def load_webqa_questions(
    data_path: Path,
    split: Optional[str] = None,
    max_items: Optional[int] = None,
) -> List[dict]:
    loader = WebQALoader(data_path)
    questions: List[dict] = []
    count = 0
    for guid, sample in loader.iter_examples():
        if split and sample.get("split") != split:
            continue
        count += 1
        if max_items is not None and count > max_items:
            break
        question = sample.get("Q")
        image_id = None
        pos_facts = sample.get("img_posFacts", [])
        if pos_facts:
            image_id = str(pos_facts[0].get("image_id") or "")
        questions.append(
            {
                "guid": guid,
                "question": question,
                "image_id": image_id,
                "answers": sample.get("A") or [],
            }
        )
    return questions


def build_webqa_relevance_map(
    data_path: Path,
    split: Optional[str] = None,
    max_items: Optional[int] = None,
) -> Dict[str, List[str]]:
    loader = WebQALoader(data_path)
    relevance: Dict[str, List[str]] = {}
    count = 0
    for guid, sample in loader.iter_examples():
        if split and sample.get("split") != split:
            continue
        count += 1
        if max_items is not None and count > max_items:
            break
        facts = list(sample.get("img_posFacts", []))
        chunk_ids = []
        for idx, fact in enumerate(facts):
            image_id = str(fact.get("image_id") or "")
            caption = str(fact.get("caption") or "").strip()
            title = str(fact.get("title") or "").strip()
            text = caption or title
            if not text:
                continue
            chunk_ids.append(f"{guid}_img_{image_id}_{idx}")
        relevance[guid] = chunk_ids
    return relevance
