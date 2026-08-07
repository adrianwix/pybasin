from __future__ import annotations

import torch
from torch import Tensor

from pybasin.regions import BallRegion, BoxRegion, Region, SphereSurfaceRegion
from pybasin.samplers.sampler import UnsupportedSamplerRegionError


def resolve_region(
    region: Region, device: str | torch.device | None, dtype: torch.dtype | None
) -> Region:
    """Move ``region``'s tensors to ``device``/``dtype``; return it unchanged if both are ``None``.

    A region only ever carries whatever device/dtype its own tensors were built with --
    this is where a sampler's explicit ``device``/``dtype`` request actually takes effect.
    """
    if device is None and dtype is None:
        return region
    if isinstance(region, BoxRegion):
        return BoxRegion(
            region.min.to(device=device, dtype=dtype), region.max.to(device=device, dtype=dtype)
        )
    if isinstance(region, SphereSurfaceRegion):
        return SphereSurfaceRegion(
            region.center.to(device=device, dtype=dtype), region.radius, region.tolerance
        )
    if isinstance(region, BallRegion):
        return BallRegion(region.center.to(device=device, dtype=dtype), region.radius)
    raise UnsupportedSamplerRegionError(
        f"Cannot resolve device/dtype for region type {type(region).__name__}."
    )


def cartesian_grid(coordinates: list[Tensor]) -> Tensor:
    if len(coordinates) == 1:
        return coordinates[0].unsqueeze(-1)
    return torch.cartesian_prod(*coordinates)


def region_bounds(region: BoxRegion | BallRegion) -> tuple[Tensor, Tensor]:
    if isinstance(region, BoxRegion):
        return region.min, region.max
    return region.center - region.radius, region.center + region.radius
