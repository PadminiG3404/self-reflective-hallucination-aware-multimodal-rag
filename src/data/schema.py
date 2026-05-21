"""Dataset schemas for multimodal ingestion."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    item_id: str
    text: str
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
