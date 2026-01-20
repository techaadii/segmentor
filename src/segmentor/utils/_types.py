from typing import Literal, Callable
from dataclasses import dataclass

import torch

type Image = torch.Tensor
type SegmentationMap = torch.Tensor
type SegmentationMask = torch.Tensor
type PixelCoordinate = tuple[int, int]


@dataclass
class Keyframe:
    embedding: torch.Tensor
    pos_features: torch.Tensor
    neg_features: torch.Tensor


type SimilarityFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


AttnImplementation = Literal[
    "sdpa", "flex_attention", "flash_attention", "flash_attention_2", "eager"
]
