from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Region(Protocol):
    """Structural interface every region (built-in or user-defined) must satisfy."""

    @property
    def state_dim(self) -> int: ...

    def contains(self, points: Tensor) -> Tensor:
        """Return a boolean membership mask for ``points``.

        :param points: Tensor with shape ``(..., state_dim)``.
        :return: Boolean tensor with shape ``points.shape[:-1]``.
        """
        ...

    @property
    def device(self) -> torch.device:
        """The device this region's own tensors currently live on."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """The dtype this region's own tensors currently have."""
        ...
