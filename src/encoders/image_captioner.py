"""Image captioning wrapper."""
from __future__ import annotations

from typing import Optional

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


class ImageCaptioner:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def caption(self, image: object, max_new_tokens: int = 30) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(output[0], skip_special_tokens=True)
        return text.strip()
