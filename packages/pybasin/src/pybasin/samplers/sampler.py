from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from pybasin.regions import Region


class UnsupportedSamplerRegionError(TypeError):
    """A sampler was asked to sample from a region type it does not support."""


class UnsupportedSamplerConfigurationError(NotImplementedError):
    """A sampler cannot run its geometry-specific algorithm for the given configuration.

    Raised for combinations that are the right region *type* but an unsupported
    parameterization of it, e.g. ``EvenlySpacedSampler`` on a ``SphereSurfaceRegion``
    with ``state_dim != 3`` (Fibonacci-sphere spacing is only defined in 3D).
    """


@runtime_checkable
class Sampler(Protocol):
    """Structural interface for samplers: ``sample(n, ...)`` returns ``n`` points."""

    region: Region
    is_deterministic: bool
    device: torch.device
    dtype: torch.dtype

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> Tensor:
        """Return exactly ``n`` initial conditions with shape ``(n, region.state_dim)``."""
        ...
