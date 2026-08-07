from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from pybasin.regions import BallRegion, BoxRegion, Region, SphereSurfaceRegion
from pybasin.samplers._utils import resolve_region
from pybasin.samplers.sampler import UnsupportedSamplerRegionError


@dataclass(init=False)
class UniformRandomSampler:
    """Uniform sampling, dispatched by region type.

    ``BoxRegion``: independent ``U(a, b)`` per active dimension.
    ``BallRegion``: uniform by *volume* (random direction, radius ``~ R * U^(1/d)``).
    ``SphereSurfaceRegion``: uniform by *surface area* (normalized Gaussian direction).
    """

    region: Region
    device: torch.device
    dtype: torch.dtype
    is_deterministic: bool

    def __init__(
        self,
        region: Region,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.region = resolve_region(region, device, dtype)
        self.device = self.region.device
        self.dtype = self.region.dtype
        self.is_deterministic = False

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> Tensor:
        region = self.region
        if isinstance(region, BoxRegion):
            return self._sample_box(region, n, generator)
        if isinstance(region, BallRegion):
            return self._sample_ball(region, n, generator)
        if isinstance(region, SphereSurfaceRegion):
            return self._sample_sphere_surface(region, n, generator)
        raise UnsupportedSamplerRegionError(
            f"UniformRandomSampler does not support region type {type(region).__name__}."
        )

    @staticmethod
    def _sample_box(region: BoxRegion, n: int, generator: torch.Generator | None) -> Tensor:
        device, dtype = region.min.device, region.min.dtype
        u = torch.rand(n, region.state_dim, device=device, dtype=dtype, generator=generator)
        return region.min + u * (region.max - region.min)

    @staticmethod
    def _sample_ball(region: BallRegion, n: int, generator: torch.Generator | None) -> Tensor:
        device, dtype = region.center.device, region.center.dtype
        d = region.state_dim
        z = torch.randn(n, d, device=device, dtype=dtype, generator=generator)
        norm = cast(Tensor, torch.linalg.vector_norm(z, dim=-1, keepdim=True))  # pyright: ignore[reportUnknownMemberType]
        directions = z / norm
        u = torch.rand(n, 1, device=device, dtype=dtype, generator=generator)
        radii = region.radius * u.pow(1.0 / d)
        return region.center + radii * directions

    @staticmethod
    def _sample_sphere_surface(
        region: SphereSurfaceRegion, n: int, generator: torch.Generator | None
    ) -> Tensor:
        device, dtype = region.center.device, region.center.dtype
        z = torch.randn(n, region.state_dim, device=device, dtype=dtype, generator=generator)
        norm = cast(Tensor, torch.linalg.vector_norm(z, dim=-1, keepdim=True))  # pyright: ignore[reportUnknownMemberType]
        directions = z / norm
        return region.center + region.radius * directions
