"""Samplers that generate complete initial conditions inside a ``Region``.

Every stochastic ``sample(n, ...)`` call returns exactly ``n`` points, each satisfying
``region.contains``. A region carries only whatever device/dtype its own tensors happen
to have; every sampler accepts an optional ``device``/``dtype`` override, applied once at
construction, that takes precedence over the region's own placement.
"""

from pybasin.samplers.cartesian_product_sampler import CartesianProductSampler
from pybasin.samplers.csv_sampler import CsvSampler
from pybasin.samplers.evenly_spaced_sampler import EvenlySpacedSampler
from pybasin.samplers.latin_hypercube_sampler import LatinHypercubeSampler
from pybasin.samplers.sampler import (
    Sampler,
    UnsupportedSamplerConfigurationError,
    UnsupportedSamplerRegionError,
)
from pybasin.samplers.sobol_sampler import SobolSampler
from pybasin.samplers.truncated_gaussian_sampler import TruncatedGaussianSampler
from pybasin.samplers.uniform_random_sampler import UniformRandomSampler

__all__ = [
    "CartesianProductSampler",
    "CsvSampler",
    "EvenlySpacedSampler",
    "LatinHypercubeSampler",
    "Sampler",
    "SobolSampler",
    "TruncatedGaussianSampler",
    "UniformRandomSampler",
    "UnsupportedSamplerConfigurationError",
    "UnsupportedSamplerRegionError",
]
