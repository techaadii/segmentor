from dataclasses import dataclass

import torch

from segmentor.utils._types import (
    Image,
    Keyframe,
    SegmentationMap,
    SimilarityFunc,
    PixelCoordinate,
)
from segmentor.utils.models.anyup import AnyUp
from segmentor.utils.models.clip import CLIPImageEncoder
from segmentor.utils.models.dinov3 import DINOv3ImageEncoder
from segmentor.utils.pipeline.dense_features import DenseFeatureEncoder
from segmentor.utils.pipeline.history import History
from segmentor.utils.pipeline.segmap import (
    probabilistic_segmentation_with_contrastive_scoring,
)
from segmentor.helpers.device import DEVICE


@dataclass
class SegmentorOutput:
    segmentation: SegmentationMap
    keyframe: Keyframe
    keyframe_sim: float
    unseen_scene: bool


class Segmentor:
    """
    The main Segmentor class.
    """

    def __init__(
        self,
        dinov3: DINOv3ImageEncoder,
        anyup: AnyUp,
        clip: CLIPImageEncoder,
        keyframe_similarity_threshold: float,
        keyframe_similarity_func: SimilarityFunc = torch.cosine_similarity,
        device: torch.device = DEVICE,
    ) -> None:
        self._dinov3 = dinov3
        self._anyup = anyup
        self._clip = clip

        self._history = History()
        self._similarity_func = keyframe_similarity_func
        self._similarity_threshold = keyframe_similarity_threshold

        self._pos_features = torch.empty(0)
        self._neg_features = torch.empty(0)

        self._dense_encoder = DenseFeatureEncoder(dinov3=dinov3, anyup=anyup)

    def register_keyframe(
        self,
        image: Image,
        pos_pixel_coords: list[PixelCoordinate],
        neg_pixel_coords: list[PixelCoordinate],
        q_chunk_size: int | None = None,
        feature_map_res: int | None = None,
        n_patches: int | None = None,
    ) -> None:
        # Load image onto device
        image = image.to(device=DEVICE)

        # Get the embedding
        with torch.inference_mode():
            embedding = self._clip(image)  # Shape: (1, D)

        # Get the dense feature map
        dense_features = self._dense_encoder(
            image,
            q_chunk_size=q_chunk_size,
            output_size=feature_map_res,
            n_patches=n_patches,
        )[0]

        # Extract positive and negative features
        pos_features = dense_features[
            :, [i for (i, _) in pos_pixel_coords], [j for (_, j) in pos_pixel_coords]
        ]
        neg_features = dense_features[
            :, [i for (i, _) in neg_pixel_coords], [j for (_, j) in neg_pixel_coords]
        ]

        # Register keyframe in history
        self._history.register_keyframe(
            embedding=embedding, pos_features=pos_features, neg_features=neg_features
        )

    def step(
        self,
        image: Image,
        q_chunk_size: int | None = None,
        output_res: tuple[int, int] | None = None,
        n_patches: int | None = None,
        gamma: float = 30,
        eta: float = 1,
        alpha: float = 1,
    ) -> SegmentorOutput | None:
        # Load image onto right device
        image = image.to(device=DEVICE)

        # See if history has anything in it
        if len(self._history) == 0:
            return None

        # Get image embedding
        with torch.inference_mode():
            embedding = self._clip(image)  # Shape: (1, D)

        # Get the best matching keyframe
        best_match_keyframe, best_similarity = self._history.search(
            query_embedding=embedding, similarity_func=self._similarity_func
        )

        # Check if new scene
        unseen_scene = best_similarity < self._similarity_threshold

        # Get dense features
        dense_features = self._dense_encoder(
            image,
            q_chunk_size=q_chunk_size,
            output_size=output_res,
            n_patches=n_patches,
        )[0]

        # Synthesize probablity map
        print(f"Feature shape: {dense_features.shape}")
        segmentation_map = probabilistic_segmentation_with_contrastive_scoring(
            featmap=dense_features,
            pos_feats=best_match_keyframe.pos_features.T,
            neg_feats=best_match_keyframe.neg_features.T,
            gamma=gamma,
            eta=eta,
            alpha=alpha,
        )

        return SegmentorOutput(
            segmentation=segmentation_map,
            keyframe=best_match_keyframe,
            keyframe_sim=best_similarity,
            unseen_scene=unseen_scene,
        )
