"""CPU/GPU backend resolution for feature selectors.

Feature selectors only need array math (variance, correlation), never a
wrapped estimator class, so unlike ``pybasin.predictors._device_backend``
this reuses :mod:`pybasin._device_backend`'s backend directly instead of
extending it -- GPU execution here only requires ``cupy``, not cuML.
"""

from pybasin._device_backend import (
    Array,
    TArray,
    from_native,
    resolve_array_backend,
    to_native,
    to_numpy,
)
from pybasin._device_backend import (
    ArrayBackend as DeviceBackend,
)

resolve_backend = resolve_array_backend

__all__ = [
    "Array",
    "DeviceBackend",
    "TArray",
    "from_native",
    "resolve_backend",
    "to_native",
    "to_numpy",
]
