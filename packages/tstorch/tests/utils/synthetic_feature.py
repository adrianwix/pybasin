# pyright: basic
"""A synthetic `Feature` class for testing the `Feature`-class test/benchmark utilities
in isolation, independent of any real calculator. Not a calculator itself - never imported
by production code, never wired into `SUPPORTED_FEATURES`. Deliberately trivial
(one shared multiply, read at each requested scale) so tests exercise the utility being
tested, not this class's own correctness.
"""

from typing import TypedDict

import torch
from torch import Tensor
from tstorch.feature_factory import Feature


class ScaleParams(TypedDict):
    scale: int


class Scale(Feature[ScaleParams]):
    """`x * scale`, one output column per requested `scale`."""

    def __init__(self, params: list[ScaleParams]) -> None:
        self.params = params
        self._scales = [p["scale"] for p in params]

    def __call__(self, x: Tensor) -> Tensor:
        return torch.cat([x[..., -1:] * s for s in self._scales], dim=-1)

    @property
    def names(self) -> list[str]:
        return [f"scale_{p['scale']}" for p in self.params]
