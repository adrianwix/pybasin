# pyright: basic
"""Backend detection for preprocessing dispatch.

Handling an eager ``torch.Tensor`` must never import NumPy or Dask - core tstorch stays
tensor-only. ``_is_dask_array`` therefore only imports Dask once the input has already
been ruled out as a ``Tensor``, so the import cost only applies to callers that actually
pass a Dask collection.
"""

from typing import Any

from torch import Tensor


def _is_tensor(value: Any) -> bool:
    """Whether ``value`` is the eager ``torch.Tensor`` backend."""
    return isinstance(value, Tensor)


def _is_dask_array(value: Any) -> bool:
    """Whether ``value`` is a ``dask.array.Array``, without importing Dask for tensors."""
    if isinstance(value, Tensor):
        return False
    import dask.array as da

    return isinstance(value, da.Array)
