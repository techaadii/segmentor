import torch
from abc import ABC, abstractmethod
from segmentor.utils._types import Image


class Encoder[TInput](torch.nn.Module, ABC):
    DIM: int

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _embed(self, x: TInput | list[TInput]) -> torch.Tensor:
        pass

    def forward(self, x: TInput | list[TInput]) -> torch.Tensor:
        return self._embed(x=x)


ImageEncoder = Encoder[Image]
TextEncoder = Encoder[str]
