import torch
from segmentor.utils._types import Keyframe, SimilarityFunc
from typing import Sized


class History(Sized):
    """The history class."""

    def __init__(self) -> None:
        self._keyframes: list[Keyframe] = []

    @property
    def keyframes(self) -> list[Keyframe]:
        return self._keyframes

    def __len__(self) -> int:
        return len(self._keyframes)

    def _get_keyframe_embeddings(self) -> torch.Tensor:
        return torch.cat([kf.embedding for kf in self._keyframes], dim=0)

    def _get_keyframe_similarities(
        self, query_embedding: torch.Tensor, similarity_func: SimilarityFunc
    ) -> torch.Tensor:
        key_embeddings = self._get_keyframe_embeddings()  # Shape: (N, D)
        return similarity_func(key_embeddings, query_embedding)

    def search(
        self,
        query_embedding: torch.Tensor,
        similarity_func: SimilarityFunc = torch.cosine_similarity,
    ) -> tuple[Keyframe, float]:
        # Get similarities with all keyframes
        similarities = self._get_keyframe_similarities(
            query_embedding=query_embedding, similarity_func=similarity_func
        )

        # Get the index of the most similar keyframe
        best_match_inx = int(similarities.flatten().argmax())

        # Get the keyframe at the best match index
        return self._keyframes[best_match_inx], similarities[best_match_inx].item()

    def register_keyframe(
        self,
        embedding: torch.Tensor,
        pos_features: torch.Tensor,
        neg_features: torch.Tensor,
    ) -> None:
        keyframe = Keyframe(
            embedding=embedding,
            pos_features=pos_features,
            neg_features=neg_features,
        )
        self._keyframes.append(keyframe)
