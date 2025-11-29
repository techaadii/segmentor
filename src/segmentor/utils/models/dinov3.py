from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModel

from segmentor.utils._types import AttnImplementation, Image
from segmentor.utils.models.encoder import ImageEncoder


class DINOv3ImageEncoder(ImageEncoder):
    """
    The DINOv3 image encoder wrapper class.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        cache_dir: Path = Path("weights/"),
        attn_implementation: AttnImplementation = "sdpa",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Load the processor and model
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_implementation,
            cache_dir=cache_dir.as_posix(),
        )

    def _embed(self, x: Image | list[Image]) -> torch.Tensor:
        # Get device
        device: torch.device = list(iter(self._model.parameters()))[0].device
        # Create inputs
        inputs = self._processor(images=x, return_tensors="pt").to(device=device)
        # Perform inference
        outputs = self._model(**inputs)

        return outputs.last_hidden_state[
            :, 1 + self._model.config.num_register_tokens :, :
        ]
