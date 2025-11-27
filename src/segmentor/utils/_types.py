from typing import Literal

import torch

type Image = torch.Tensor
type SegmentationMap = torch.Tensor
type SegmentationMask = torch.Tensor

AttnImplementation = Literal[
    "sdpa", "flex_attention", "flash_attention", "flash_attention_2"
]
