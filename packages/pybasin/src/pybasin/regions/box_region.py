from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from pybasin.regions._utils import as_tensor


@dataclass(frozen=True, init=False)
class BoxRegion:
    """An axis-aligned box. A dimension is fixed when ``min[j] == max[j]``."""

    min: Tensor
    max: Tensor

    def __init__(self, min: Tensor | Sequence[float], max: Tensor | Sequence[float]) -> None:
        """
        :param min: Per-axis lower bound, as a ``Tensor`` or a plain sequence of floats
            (converted to a float32 CPU tensor; use a ``Tensor`` directly for another
            device/dtype).
        :param max: Per-axis upper bound, same shape rules as ``min``.
        """
        min_t, max_t = as_tensor(min), as_tensor(max)
        if min_t.ndim != 1:
            raise ValueError(f"min must be one-dimensional, got shape {tuple(min_t.shape)}.")
        if max_t.ndim != 1:
            raise ValueError(f"max must be one-dimensional, got shape {tuple(max_t.shape)}.")
        if min_t.shape != max_t.shape:
            raise ValueError(
                f"min and max must have the same shape, got "
                f"{tuple(min_t.shape)} and {tuple(max_t.shape)}."
            )
        if min_t.dtype != max_t.dtype:
            raise ValueError(
                f"min and max must share a dtype, got {min_t.dtype} and {max_t.dtype}."
            )
        if min_t.device != max_t.device:
            raise ValueError(
                f"min and max must share a device, got {min_t.device} and {max_t.device}."
            )
        if bool((min_t > max_t).any()):
            raise ValueError("min must be <= max elementwise.")
        object.__setattr__(self, "min", min_t)
        object.__setattr__(self, "max", max_t)

    @property
    def state_dim(self) -> int:
        return self.min.numel()

    @property
    def device(self) -> torch.device:
        return self.min.device

    @property
    def dtype(self) -> torch.dtype:
        return self.min.dtype

    def contains(self, points: Tensor) -> Tensor:
        return ((points >= self.min) & (points <= self.max)).all(dim=-1)

    @property
    def fixed_dimensions(self) -> Tensor:
        """Boolean mask, ``True`` where ``min[j] == max[j]``."""
        return self.min == self.max

    @property
    def active_dimensions(self) -> Tensor:
        """Boolean mask, ``True`` where the dimension has a nonzero extent."""
        return self.min != self.max
