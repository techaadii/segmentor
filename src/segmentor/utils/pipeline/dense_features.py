import torch
from segmentor.utils.models.encoder import ImageEncoder
from segmentor.utils.models.anyup import AnyUp
from segmentor.utils._types import Image
from typing_extensions import override


class DenseFeatureEncoder(torch.nn.Module):
    """
    The dense feature encoder module concatenating:
    1. Image encoder for patch-level features
    2. AnyUp feature upsampler for fine-grained pixel-level features
    """

    def __init__(
        self, image_encoder: ImageEncoder, anyup: AnyUp, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._image_encoder = image_encoder
        self._anyup = anyup

    @override
    def forward(
        self,
        image: Image,
        q_chunk_size: int | None = None,
        output_size: tuple[int, int] | None = None,
        n_patches: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # Get patch features
        with torch.inference_mode():
            patch_features: torch.Tensor = self._image_encoder(x=image)

        # Pass through the AnyUp feature upsampler
        with torch.inference_mode():
            dense_features: torch.Tensor = self._anyup(
                image=image.to(dtype=torch.float32),
                features=patch_features.to(dtype=torch.float32),
                q_chunk_size=q_chunk_size,
                output_size=output_size,
                n_patches=n_patches,
            )

        return dense_features
