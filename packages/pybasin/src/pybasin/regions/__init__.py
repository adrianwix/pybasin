"""Geometric regions of interest for initial-condition sampling.

A region defines only geometric membership (``contains``) and the boundary numbers
themselves. It has no opinion on device/dtype placement or how points inside it are
generated -- pass ``device``/``dtype`` to a sampler (see ``pybasin.samplers``) for that.
"""

from pybasin.regions.ball_region import BallRegion
from pybasin.regions.box_region import BoxRegion
from pybasin.regions.region import Region
from pybasin.regions.sphere_surface_region import SphereSurfaceRegion

__all__ = [
    "BallRegion",
    "BoxRegion",
    "Region",
    "SphereSurfaceRegion",
]
