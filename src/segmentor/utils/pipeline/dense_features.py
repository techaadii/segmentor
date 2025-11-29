import torch
from segmentor.utils.models.dinov3 import DINOv3ImageEncoder
from segmentor.utils.models.anyup import AnyUp
from segmentor.utils._types import Image


class DenseFeatureEncoder(torch.nn.Module):
    """
    The dense feature encoder module concatenating:
    1. DINOv3 image encoder for coarse features
    2. AnyUp feature upsampler for fine-grained pixel-wise features
    """

    def __init__(
        self, dinov3: DINOv3ImageEncoder, anyup: AnyUp, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._dinov3 = dinov3
        self._anyup = anyup

    def forward(
        self,
        image: Image,
        q_chunk_size: int | None = None,
        output_size: tuple[int, int] | None = None,
        n_patches: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # Get DINOv3 features
        with torch.inference_mode():
            dinov3_features: torch.Tensor = self._dinov3(x=image)

        # Pass through the AnyUp feature upsampler
        with torch.inference_mode():
            dense_features: torch.Tensor = self._anyup(
                image=image.to(dtype=torch.float32),
                features=dinov3_features.to(dtype=torch.float32),
                q_chunk_size=q_chunk_size,
                output_size=output_size,
                n_patches=n_patches,
            )

        return dense_features
