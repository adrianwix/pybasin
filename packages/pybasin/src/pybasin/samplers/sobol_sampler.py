from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from pybasin.regions import BoxRegion, Region
from pybasin.samplers._utils import resolve_region
from pybasin.samplers.sampler import UnsupportedSamplerRegionError


@dataclass(init=False)
class SobolSampler:
    """Sobol quasi-random sampling over active ``BoxRegion`` dimensions.

    ``scramble=True`` (the default) is recommended for general use; unscrambled Sobol
    points share pathological correlations along axis-aligned slices. Batch sizes that
    are powers of two tend to be the most balanced.
    """

    region: Region
    scramble: bool
    seed: int | None
    device: torch.device
    dtype: torch.dtype
    is_deterministic: bool

    def __init__(
        self,
        region: Region,
        scramble: bool = True,
        seed: int | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.region = resolve_region(region, device, dtype)
        self.scramble = scramble
        self.seed = seed
        self.device = self.region.device
        self.dtype = self.region.dtype
        self.is_deterministic = False
        if not isinstance(self.region, BoxRegion):
            raise UnsupportedSamplerRegionError(
                f"SobolSampler supports BoxRegion only, got {type(self.region).__name__}."
            )

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> Tensor:
        region = cast(BoxRegion, self.region)
        device, dtype = region.min.device, region.min.dtype
        active_idx = torch.nonzero(region.active_dimensions, as_tuple=True)[0]

        points = region.min.unsqueeze(0).repeat(n, 1).clone()
        n_active = int(active_idx.numel())
        if n_active == 0:
            return points

        engine = SobolEngine(dimension=n_active, scramble=self.scramble, seed=self.seed)
        unit = engine.draw(n).to(device=device, dtype=dtype)

        lo, hi = region.min[active_idx], region.max[active_idx]
        points[:, active_idx] = lo + unit * (hi - lo)
        return points
