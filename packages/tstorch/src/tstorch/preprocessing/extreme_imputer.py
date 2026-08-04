# pyright: basic
"""Elementwise finite-bound imputation, independent of any extractor."""

from typing import Any

import torch
from torch import Tensor

from tstorch._array_backend import array_to_tensor, tensor_to_array_like
from tstorch.preprocessing._dispatch import _is_dask_array, _is_tensor


def _extreme_bound(dtype: torch.dtype) -> float:
    """Largest safe finite bound for ``dtype``.

    ``finfo(dtype).max ** 0.25`` sits well below ``finfo(dtype).max`` itself, so the
    returned bound is a large, distinguishable value rather than the dtype's own overflow
    edge. Scales with the dtype's own headroom: for example ``~16`` for float16, ``~4.29e9``
    for float32, ``~1.16e77`` for float64.
    """
    finfo = torch.finfo(dtype)
    return finfo.max**0.25


class ExtremeImputer:
    """Replace non-finite feature values with large, sign-preserving bounds.

    Stateless and elementwise: every value is imputed independently of every other, so
    local and distributed (block-by-block) results agree exactly. NaN and positive
    infinity map to the positive bound; negative infinity maps to the negative
    bound. Useful when samples with non-finite features (e.g. unbounded trajectories)
    should remain distinguishable rather than being folded into the finite population.

    :param bound: Fixed bound magnitude to use for every dtype, overriding the
        default per-dtype value from :func:`_extreme_bound`. ``None`` uses the default.
    :raises TypeError: If ``transform`` receives a non-floating or complex tensor.
    """

    def __init__(self, bound: float | None = None) -> None:
        self._bound = bound

    def transform(self, features: Any) -> Any:
        """Impute ``features`` in place of NaN/inf, preserving dtype and device.

        :param features: Eager ``torch.Tensor`` of any shape, or a NumPy- or CuPy-backed
            ``dask.array.Array`` of any shape.
        :return: The same collection family and backend, same shape and dtype, with every
            NaN and infinite value replaced by ``+bound`` or ``-bound``. Imputation is
            elementwise, so a Dask result is identical to imputing the same array eagerly.
            A CuPy-backed input stays CuPy-backed - every block round-trips through a CUDA
            tensor without touching the host.
        """
        if _is_tensor(features):
            return _transform_tensor(features, self._bound)
        if _is_dask_array(features):
            return features.map_blocks(
                _transform_tensor_as_array,
                bound=self._bound,
                dtype=features.dtype,
                meta=features._meta,
            )
        raise TypeError(
            f"ExtremeImputer requires a torch.Tensor or dask.array.Array, got {type(features)}"
        )


def _transform_tensor(features: Tensor, bound: float | None) -> Tensor:
    if features.is_complex() or not features.is_floating_point():
        raise TypeError(
            f"ExtremeImputer requires a real floating tensor, got dtype={features.dtype}"
        )

    resolved_bound = bound if bound is not None else _extreme_bound(features.dtype)
    positive = torch.tensor(resolved_bound, device=features.device, dtype=features.dtype)
    negative = torch.tensor(-resolved_bound, device=features.device, dtype=features.dtype)

    result = torch.where(torch.isneginf(features), negative, features)
    result = torch.where(torch.isposinf(result), positive, result)
    result = torch.where(torch.isnan(result), positive, result)
    return result


def _transform_tensor_as_array(block: Any, bound: float | None) -> Any:
    """Impute one NumPy or CuPy block by round-tripping through the tensor transform."""
    tensor = array_to_tensor(block)
    return tensor_to_array_like(_transform_tensor(tensor, bound), like=block)
