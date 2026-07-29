"""Constants used throughout the pybasin package."""

DEFAULT_CACHE_DIR: str = ".pybasin_cache"
"""Default directory for caching integration results."""

DEFAULT_STEADY_FRACTION: float = 0.85
"""Default fraction of time span to skip when filtering transient dynamics.
A value of 0.85 means only the last 15% of the time series is used for
steady-state feature extraction and orbit data computation."""

UNSET = object()
"""Sentinel to distinguish 'not provided' from ``None`` in optional keyword arguments."""
