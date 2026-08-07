from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from pybasin.regions import BallRegion, BoxRegion, Region
from pybasin.samplers._utils import cartesian_grid, region_bounds, resolve_region
from pybasin.samplers.sampler import UnsupportedSamplerRegionError


@dataclass(init=False)
class CartesianProductSampler:
    """Builds the Cartesian product of explicit per-dimension coordinate tensors.

    Each dimension gets its own coordinate tensor directly, independent of the region's
    own bounds: ``torch.arange(0, 10.01, 0.01)`` for one dimension and
    ``torch.arange(100, 1001, 1)`` for another. Afterward, the product is filtered by
    ``region.contains``, since user-supplied coordinates aren't guaranteed to respect the
    region.

    Unlike every other sampler here, ``sample`` takes no ``n`` at all; the point count is
    entirely determined by ``coordinates``, and there's no meaningful "first n" of a
    Cartesian grid to slice. Call ``n_points()`` to check the resulting count up front,
    without generating anything, for when you want to size ``coordinates`` before calling
    ``sample``.
    """

    region: Region
    coordinates: Sequence[Tensor]
    device: torch.device
    dtype: torch.dtype
    is_deterministic: bool

    def __init__(
        self,
        region: Region,
        coordinates: Sequence[Tensor],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.region = resolve_region(region, device, dtype)
        self.coordinates = coordinates
        self.device = self.region.device
        self.dtype = self.region.dtype
        self.is_deterministic = True

        region = self.region
        if not isinstance(region, (BoxRegion, BallRegion)):
            raise UnsupportedSamplerRegionError(
                "CartesianProductSampler supports BoxRegion and BallRegion only, got "
                f"{type(region).__name__}."
            )
        if len(self.coordinates) != region.state_dim:
            raise ValueError(
                f"len(coordinates) must equal state_dim ({region.state_dim}), "
                f"got {len(self.coordinates)}."
            )
        for j, c in enumerate(self.coordinates):
            if c.ndim != 1:
                raise ValueError(f"coordinates[{j}] must be one-dimensional, got shape {c.shape}.")

    def n_points(self) -> int:
        """Total points in the raw Cartesian product, before region filtering."""
        size = 1
        for c in self.coordinates:
            size *= c.numel()
        return size

    def sample(self) -> Tensor:
        region = cast(BoxRegion | BallRegion, self.region)
        min_bound, _ = region_bounds(region)
        coordinates = [
            c.to(device=min_bound.device, dtype=min_bound.dtype) for c in self.coordinates
        ]
        grid = cartesian_grid(coordinates)
        return grid[region.contains(grid)]
