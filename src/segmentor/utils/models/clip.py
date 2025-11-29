import torch
from segmentor.utils.models.encoder import ImageEncoder
from segmentor.utils._types import Image, AttnImplementation
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from pathlib import Path


class CLIPImageEncoder(ImageEncoder):
    """The CLIP image encoder wrapper class."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: Path = Path("weights/"),
        attn_implementation: AttnImplementation = "sdpa",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self._model = CLIPVisionModelWithProjection.from_pretrained(
            model_name, attn_implementation=attn_implementation, cache_dir=cache_dir
        )

    def _embed(self, x: Image | list[Image]) -> torch.Tensor:
        inputs = self._processor(images=x, return_tensors="pt")
        outputs = self._model(**inputs)
        return outputs.image_embeds
