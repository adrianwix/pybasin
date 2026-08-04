# pyright: basic
"""Backend-agnostic NumPy/CuPy array detection and conversion.

The only place that distinguishes a NumPy from a CuPy array. Every Dask code path that
touches a raw array block - worker feature extraction, preprocessing over a Dask Array -
goes through this module instead of assuming NumPy, so a CuPy or ``dask-cudf``-backed
collection stays GPU-resident end to end rather than round-tripping through the host.
"""

from typing import Any

import torch
from torch import Tensor


def is_cupy_array(value: Any) -> bool:
    """Whether ``value`` is a ``cupy.ndarray``, without importing CuPy for NumPy arrays."""
    try:
        import cupy
    except ImportError:
        return False
    return isinstance(value, cupy.ndarray)


def array_to_tensor(array: Any) -> Tensor:
    """Convert a NumPy or CuPy array into a ``torch.Tensor``.

    A CuPy array converts via DLPack onto the CUDA device it already lives on; a NumPy
    array converts via the standard zero-copy CPU view.
    """
    if is_cupy_array(array):
        return torch.from_dlpack(array)
    return torch.from_numpy(array)


def tensor_to_array_like(tensor: Tensor, like: Any) -> Any:
    """Convert ``tensor`` to the same NumPy/CuPy backend as ``like``.

    Moves ``tensor`` onto CUDA first when ``like`` is a CuPy array, or onto the CPU when
    ``like`` is a NumPy array, regardless of which device ``tensor`` already sits on.
    """
    if is_cupy_array(like):
        import cupy

        return cupy.from_dlpack(tensor.detach().to("cuda"))
    return tensor.detach().cpu().numpy()
