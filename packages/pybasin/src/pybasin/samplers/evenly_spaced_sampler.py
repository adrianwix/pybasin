from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from pybasin.regions import BallRegion, BoxRegion, Region, SphereSurfaceRegion
from pybasin.samplers._utils import cartesian_grid, region_bounds, resolve_region
from pybasin.samplers.sampler import (
    UnsupportedSamplerConfigurationError,
    UnsupportedSamplerRegionError,
)


@dataclass(init=False)
class EvenlySpacedSampler:
    """Deterministic, geometry-aware coverage, dispatched by region type.

    ``sample(n)`` works like every other sampler. For ``BoxRegion``/``BallRegion``, ``n``
    is distributed across active dimensions the same way the legacy ``GridSampler`` did
    (``ceil(n ** (1 / n_active))`` per axis), so the actual count is approximately ``n``
    (``BallRegion`` filters the enclosing-box grid by ``contains`` afterward, so its count
    can additionally fall below that approximation -- use ``n_points(n)`` to check the
    exact resulting count before calling ``sample``). ``SphereSurfaceRegion`` requires
    ``state_dim == 3`` and uses ``n`` directly as the exact Fibonacci-sphere point count.
    For an exact, non-approximate grid built from explicit per-axis coordinates instead of
    an ``n`` target, use ``CartesianProductSampler``.
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
        self.is_deterministic = True

        region = self.region
        if isinstance(region, SphereSurfaceRegion):
            if region.state_dim != 3:
                raise UnsupportedSamplerConfigurationError(
                    "Fibonacci-sphere spacing is only implemented for state_dim == 3, "
                    f"got state_dim == {region.state_dim}."
                )
        elif not isinstance(region, (BoxRegion, BallRegion)):
            raise UnsupportedSamplerRegionError(
                f"EvenlySpacedSampler does not support region type {type(region).__name__}."
            )

    def sample(self, n: int, *, seed: int | None = None) -> Tensor:
        region = self.region
        if isinstance(region, SphereSurfaceRegion):
            if n < 1:
                raise ValueError(f"n must be >= 1, got {n}.")
            return self._sample_fibonacci_sphere(region, n)

        region = cast(BoxRegion | BallRegion, region)
        shape = self._auto_grid_shape(region, n)
        grid = cartesian_grid(self._grid_coordinates(*region_bounds(region), shape))
        if isinstance(region, BallRegion):
            return grid[region.contains(grid)]
        return grid

    def n_points(self, n: int) -> int:
        """The exact grid size ``sample(n)`` would build, before any ``BallRegion`` filtering."""
        region = self.region
        if isinstance(region, SphereSurfaceRegion):
            return n
        region = cast(BoxRegion | BallRegion, region)
        shape = self._auto_grid_shape(region, n)
        size = 1
        for count in shape:
            size *= count
        return size

    @staticmethod
    def _grid_coordinates(
        min_bound: Tensor, max_bound: Tensor, shape: tuple[int, ...]
    ) -> list[Tensor]:
        return [
            torch.linspace(
                lo.item(), hi.item(), count, device=min_bound.device, dtype=min_bound.dtype
            )
            for lo, hi, count in zip(min_bound, max_bound, shape, strict=True)
        ]

    @staticmethod
    def _auto_grid_shape(region: BoxRegion | BallRegion, n: int) -> tuple[int, ...]:
        """Per-axis point counts that multiply out to approximately ``n``.

        Matches how many points naively fit a Cartesian grid: distribute ``n`` evenly
        across active dimensions (``ceil(n ** (1 / n_active))`` per axis), so the actual
        total is usually slightly above ``n``, never below it. Fixed ``BoxRegion``
        dimensions always get a count of 1; ``BallRegion`` has no fixed-dimension concept,
        so every axis is active.
        """
        if isinstance(region, BoxRegion):
            fixed = cast(list[bool], region.fixed_dimensions.tolist())  # pyright: ignore[reportUnknownMemberType]
        else:
            fixed = [False] * region.state_dim
        n_active = sum(not is_fixed for is_fixed in fixed)
        if n_active == 0:
            return tuple(1 for _ in fixed)
        per_dim = max(1, math.ceil(n ** (1 / n_active)))
        return tuple(1 if is_fixed else per_dim for is_fixed in fixed)

    @staticmethod
    def _sample_fibonacci_sphere(region: SphereSurfaceRegion, num_points: int) -> Tensor:
        device, dtype = region.center.device, region.center.dtype
        i = torch.arange(num_points, device=device, dtype=dtype)
        z = 1 - 2 * (i + 0.5) / num_points
        golden_angle = math.pi * (3 - math.sqrt(5))
        phi = i * golden_angle
        radius_xy = torch.sqrt(torch.clamp(1 - z * z, min=0))
        x = radius_xy * torch.cos(phi)
        y = radius_xy * torch.sin(phi)
        unit_points = torch.stack([x, y, z], dim=-1)
        return region.center + region.radius * unit_points
