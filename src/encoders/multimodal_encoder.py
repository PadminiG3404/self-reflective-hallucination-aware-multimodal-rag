"""Multimodal encoder using CLIP or BLIP-2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Blip2Model, Blip2Processor, CLIPModel, CLIPProcessor


@dataclass
class EncoderOutput:
    embeddings: torch.Tensor
    pooled: Optional[torch.Tensor] = None


class MultimodalEncoder:
    def __init__(
        self,
        model_type: str,
        model_name: str,
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.model_type == "clip":
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        elif self.model_type == "blip2":
            self._model = Blip2Model.from_pretrained(self.model_name).to(self.device)
            self._processor = Blip2Processor.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        self._model.eval()

    def encode_image(self, images: List[Image.Image]) -> EncoderOutput:
        self._load()
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.model_type == "clip":
                embeddings = self._model.get_image_features(**inputs)
            else:
                embeddings = self._model.get_image_features(**inputs)
        embeddings = self._unwrap_embeddings(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return EncoderOutput(embeddings=embeddings)

    def encode_text(self, texts: List[str]) -> EncoderOutput:
        self._load()
        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            if self.model_type == "clip":
                embeddings = self._model.get_text_features(**inputs)
            else:
                embeddings = self._model.get_text_features(**inputs)
        embeddings = self._unwrap_embeddings(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return EncoderOutput(embeddings=embeddings)

    def cross_modal_similarity(
        self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.normalize:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        return image_embeddings @ text_embeddings.t()

    @staticmethod
    def _unwrap_embeddings(embeddings: torch.Tensor | object) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        pooler_output = getattr(embeddings, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        last_hidden_state = getattr(embeddings, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state[:, 0]
        raise TypeError("Unsupported embedding output type")
