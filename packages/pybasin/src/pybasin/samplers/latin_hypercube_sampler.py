from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from pybasin.regions import BoxRegion, Region
from pybasin.samplers._utils import resolve_region
from pybasin.samplers.sampler import UnsupportedSamplerRegionError


@dataclass(init=False)
class LatinHypercubeSampler:
    """Latin hypercube sampling over active ``BoxRegion`` dimensions; fixed dims are inserted after."""

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
        if not isinstance(self.region, BoxRegion):
            raise UnsupportedSamplerRegionError(
                f"LatinHypercubeSampler supports BoxRegion only, got {type(self.region).__name__}."
            )

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> Tensor:
        region = cast(BoxRegion, self.region)
        device, dtype = region.min.device, region.min.dtype
        active_idx = torch.nonzero(region.active_dimensions, as_tuple=True)[0]

        points = region.min.unsqueeze(0).repeat(n, 1).clone()
        n_active = int(active_idx.numel())
        if n_active == 0:
            return points

        interval = torch.arange(n, device=device, dtype=dtype).unsqueeze(1).expand(n, n_active)
        permutation = torch.argsort(
            torch.rand(n, n_active, device=device, generator=generator), dim=0
        )
        permuted_interval = torch.gather(interval, 0, permutation)
        offset = torch.rand(n, n_active, device=device, dtype=dtype, generator=generator)
        unit = (permuted_interval + offset) / n

        lo, hi = region.min[active_idx], region.max[active_idx]
        points[:, active_idx] = lo + unit * (hi - lo)
        return points
