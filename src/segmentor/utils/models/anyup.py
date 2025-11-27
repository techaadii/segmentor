import torch
from math import sqrt

from segmentor.utils._types import Image


class AnyUp(torch.nn.Module):
    """AnyUp feature upsampler wrapper class"""

    GITHUB_REPO = "wimmerth/anyup"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Load the model from GitHub
        self._model: torch.nn.Module = torch.hub.load(
            AnyUp.GITHUB_REPO, "anyup_multi_backbone", use_natten=True
        )  # type: ignore

    @staticmethod
    def prepare_features(
        features: torch.Tensor, n_patches: int | tuple[int, int] | None = None
    ) -> torch.Tensor:
        """
        Utility function to reshape low-resolution image encoder output
        features.
        `(B, C, D) -> (B, D, h, w)`, where `C = h * w`

        Args:
            features (torch.Tensor): The low-resolution features.
                Shape: `(B, C, D)`
            n_patches (Optional, int | tuple[int, int]): The patch size.
                - `int`: `h = w = patch_size`
                - `tuple[int, int]`: `h, w = patch_size`
                - `None`: `h = w = sqrt(C)`

        Returns:
            torch.Tensor:
                The patch features prepared for input into the AnyUp
                feature upsampler. Shape: `(B, D, h, w)`

        ## Note:
            - `B` = Batch size
            - `D` = Feature dimensionality
            - `h` = Height of feature map. `h = H / patch_size`
            - `w` = Width of feature map. `w = W / patch_size`
            - The original image had a resolution `W x H` (e.g. `1920 x 1080`)
        """
        _, C, _ = features.shape

        if type(n_patches) is int:
            h = w = n_patches
        elif type(n_patches) is tuple:
            h, w = n_patches
        else:
            h = w = int(sqrt(C))

        return features.permute(0, 2, 1).unflatten(dim=-1, sizes=(h, w))

    def forward(
        self,
        image: Image,
        features: torch.Tensor,
        output_size: tuple[int, int] | None = None,
        q_chunk_size: int | None = None,
        n_patches: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # Preprocess features
        feature_map = AnyUp.prepare_features(features=features, n_patches=n_patches)

        # Output size
        if output_size is None:
            output_size = image.shape[-2:]  # type: ignore

        return self._model(
            image, feature_map, output_size=output_size, q_chunk_size=q_chunk_size
        )
