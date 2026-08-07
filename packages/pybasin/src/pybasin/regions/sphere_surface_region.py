from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from pybasin.regions._utils import as_tensor


@dataclass(frozen=True, init=False)
class SphereSurfaceRegion:
    """The surface of a sphere, ``{x : | ||x - center|| - radius | <= tolerance}``."""

    center: Tensor
    radius: float
    tolerance: float

    def __init__(
        self, center: Tensor | Sequence[float], radius: float, tolerance: float = 1e-6
    ) -> None:
        """
        :param center: Sphere center, as a ``Tensor`` or a plain sequence of floats
            (converted to a float32 CPU tensor; use a ``Tensor`` directly for another
            device/dtype).
        :param radius: Sphere radius, must be ``>= 0``.
        :param tolerance: Shell half-thickness accepted by ``contains``, must be ``>= 0``.
        """
        center_t = as_tensor(center)
        if center_t.ndim != 1:
            raise ValueError(f"center must be one-dimensional, got shape {tuple(center_t.shape)}.")
        if radius < 0:
            raise ValueError(f"radius must be >= 0, got {radius}.")
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}.")
        object.__setattr__(self, "center", center_t)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def state_dim(self) -> int:
        return self.center.numel()

    @property
    def device(self) -> torch.device:
        return self.center.device

    @property
    def dtype(self) -> torch.dtype:
        return self.center.dtype

    def contains(self, points: Tensor) -> Tensor:
        distance = cast(
            Tensor,
            torch.linalg.vector_norm(points - self.center, dim=-1),  # pyright: ignore[reportUnknownMemberType]
        )
        return torch.abs(distance - self.radius) <= self.tolerance
