from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pybasin.regions import BallRegion, BoxRegion, Region
from pybasin.samplers._utils import resolve_region
from pybasin.samplers.sampler import UnsupportedSamplerRegionError


@dataclass(init=False)
class TruncatedGaussianSampler:
    """Gaussian candidates, kept only when ``region.contains`` accepts them (never clipped).

    Clipping would pile artificial mass on the boundary; rejection preserves the true
    truncated-Gaussian shape. Supported regions: ``BoxRegion``, ``BallRegion``.
    """

    region: Region
    mean: Tensor
    std: Tensor
    candidate_batch_size: int | None
    max_candidate_batches: int
    seed: int | None
    device: torch.device
    dtype: torch.dtype
    is_deterministic: bool

    def __init__(
        self,
        region: Region,
        mean: Tensor,
        std: Tensor,
        candidate_batch_size: int | None = None,
        max_candidate_batches: int = 1000,
        seed: int | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.region = resolve_region(region, device, dtype)
        self.device = self.region.device
        self.dtype = self.region.dtype
        self.mean = mean.to(device=self.device, dtype=self.dtype)
        self.std = std.to(device=self.device, dtype=self.dtype)
        self.candidate_batch_size = candidate_batch_size
        self.max_candidate_batches = max_candidate_batches
        self.seed = seed
        self.is_deterministic = False

        region = self.region
        if not isinstance(region, (BoxRegion, BallRegion)):
            raise UnsupportedSamplerRegionError(
                "TruncatedGaussianSampler supports BoxRegion and BallRegion only, got "
                f"{type(region).__name__}."
            )
        if isinstance(region, BoxRegion):
            fixed = region.fixed_dimensions
            if bool((self.mean[fixed] != region.min[fixed]).any()):
                raise ValueError("mean must equal the fixed value on every fixed box dimension.")
            if bool((self.std[fixed] != 0).any()):
                raise ValueError("std must be zero on every fixed box dimension.")

    def sample(self, n: int, *, seed: int | None = None) -> Tensor:
        effective_seed = seed if seed is not None else self.seed
        generator = (
            torch.Generator(device=self.device).manual_seed(effective_seed)
            if effective_seed is not None
            else None
        )
        region = self.region
        batch_size = self.candidate_batch_size or max(n, 1)
        collected: list[Tensor] = []
        collected_count = 0
        for _ in range(self.max_candidate_batches):
            if collected_count >= n:
                break
            candidates = self.mean + self.std * torch.randn(
                batch_size,
                region.state_dim,
                device=self.mean.device,
                dtype=self.mean.dtype,
                generator=generator,
            )
            accepted = candidates[region.contains(candidates)]
            if accepted.numel():
                collected.append(accepted)
                collected_count += accepted.shape[0]
        if collected_count < n:
            raise RuntimeError(
                f"TruncatedGaussianSampler collected only {collected_count}/{n} points within "
                f"{self.max_candidate_batches} candidate batches."
            )
        return torch.cat(collected, dim=0)[:n]
