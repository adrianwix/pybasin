# pyright: basic
"""Global standardization, independent of any extractor."""

from typing import Any, NamedTuple

import torch
from torch import Tensor

from tstorch._array_backend import array_to_tensor, tensor_to_array_like
from tstorch.preprocessing._dispatch import _is_dask_array, _is_tensor

_MIN_ARITHMETIC_DTYPE = torch.float32


class ScalingState(NamedTuple):
    """Fitted statistics produced by ``StandardScaler.fit``.

    Identical in shape whether the fit ran eagerly on a ``torch.Tensor`` or over a
    distributed Dask Array: ``mean`` and ``scale`` are always one-dimensional tensors, so
    no caller branches on which backend produced them. Device follows the fit input:
    fitting on a CUDA tensor or a CuPy-backed Dask Array keeps ``mean``/``scale`` on that
    GPU (a GPU pipeline never touches the host); fitting on a CPU tensor or a NumPy-backed
    Dask Array produces CPU tensors.

    :ivar mean: Per-feature mean, tensor of shape ``(F,)``.
    :ivar scale: Per-feature population standard deviation, tensor of shape ``(F,)``.
        Zero-variance columns are given a scale of 1 so ``transform`` never divides by
        zero.
    :ivar n_samples_seen: Number of rows the fit was computed over.
    """

    mean: Tensor
    scale: Tensor
    n_samples_seen: int


def _promoted_dtype(dtype: torch.dtype) -> torch.dtype:
    """float16/bfloat16 promote to float32 for safe arithmetic; other dtypes pass through."""
    if dtype in (torch.float16, torch.bfloat16):
        return _MIN_ARITHMETIC_DTYPE
    return dtype


class StandardScaler:
    """Fit and apply one global mean/variance standardization.

    ``fit`` computes one mean and variance per feature across the *complete* training
    collection - no partition or worker fits itself independently - and uses population
    variance (``ddof=0``), matching Dask-ML's ``StandardScaler`` and scikit-learn's
    default so torch, scikit-learn, and Dask-ML paths agree. ``transform`` then applies
    those same statistics to every row/partition. The fitted :class:`ScalingState` follows
    the input's own backend and device: an eager CUDA tensor or a CuPy-backed Dask Array
    keeps ``mean``/``scale`` on that GPU, so a GPU pipeline never touches the host; a
    NumPy-backed Dask Array or CPU tensor produces a CPU-resident state.

    :raises RuntimeError: If ``state`` is accessed before ``fit``.
    """

    def __init__(self) -> None:
        self._state: ScalingState | None = None

    def fit(self, features: Any) -> "StandardScaler":
        """Compute one mean and population variance per feature (column).

        :param features: Eager ``torch.Tensor`` of shape ``(B, F)``, or a NumPy- or
            CuPy-backed ``dask.array.Array`` of shape ``(B, F)``.
        :return: ``self``, now fitted.
        """
        if _is_tensor(features):
            self._state = _fit_tensor(features)
        elif _is_dask_array(features):
            self._state = _fit_dask_array(features)
        else:
            raise TypeError(
                f"StandardScaler.fit requires a torch.Tensor or dask.array.Array, got {type(features)}"
            )
        return self

    def transform(self, features: Any) -> Any:
        """Apply the fitted mean/scale to ``features``.

        :param features: Same collection family accepted by ``fit``.
        :return: The same collection family, with each column standardized to zero mean
            and unit variance under the fitted statistics. float16/bfloat16 input is
            promoted to float32 to keep centering and scaling arithmetic finite.
        :raises RuntimeError: If called before ``fit``.
        """
        if self._state is None:
            raise RuntimeError("StandardScaler.transform called before fit")

        if _is_tensor(features):
            return _transform_tensor(features, self._state)
        if _is_dask_array(features):
            return _transform_dask_array(features, self._state)
        raise TypeError(
            f"StandardScaler.transform requires a torch.Tensor or dask.array.Array, got {type(features)}"
        )

    @property
    def state(self) -> ScalingState:
        """The fitted :class:`ScalingState`.

        :raises RuntimeError: If accessed before ``fit``.
        """
        if self._state is None:
            raise RuntimeError("StandardScaler.state accessed before fit")
        return self._state


def _fit_tensor(features: Tensor) -> ScalingState:
    compute_dtype = _promoted_dtype(features.dtype)
    values = features.to(compute_dtype)
    mean = values.mean(dim=0)
    variance = values.var(dim=0, unbiased=False)
    scale = torch.sqrt(variance)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return ScalingState(
        mean=mean.detach(),
        scale=scale.detach(),
        n_samples_seen=features.shape[0],
    )


def _transform_tensor(features: Tensor, state: ScalingState) -> Tensor:
    compute_dtype = _promoted_dtype(features.dtype)
    values = features.to(compute_dtype)
    mean = state.mean.to(device=features.device, dtype=compute_dtype)
    scale = state.scale.to(device=features.device, dtype=compute_dtype)
    return (values - mean) / scale


def _fit_dask_array(features: Any) -> ScalingState:
    from dask_ml.preprocessing import StandardScaler as _DaskMLStandardScaler

    n_samples = features.shape[0]
    compute_dtype = _promoted_np_dtype(features.dtype)

    dask_ml_scaler = _DaskMLStandardScaler()
    dask_ml_scaler.fit(features.astype(compute_dtype))

    # `mean_`/`scale_` come back as whatever array backend `features` used - NumPy for a
    # NumPy-backed Dask Array, CuPy for a CuPy-backed one - since Dask-ML computes them
    # via the array's own reductions rather than forcing a NumPy conversion.
    return ScalingState(
        mean=array_to_tensor(dask_ml_scaler.mean_).detach(),
        scale=array_to_tensor(dask_ml_scaler.scale_).detach(),
        n_samples_seen=n_samples,
    )


def _transform_dask_array(features: Any, state: ScalingState) -> Any:
    compute_dtype = _promoted_np_dtype(features.dtype)
    mean = tensor_to_array_like(state.mean, like=features._meta).astype(compute_dtype)
    scale = tensor_to_array_like(state.scale, like=features._meta).astype(compute_dtype)
    return (features.astype(compute_dtype) - mean) / scale


def _promoted_np_dtype(dtype: Any) -> Any:
    import numpy as np

    if dtype == np.dtype("float16"):
        return np.dtype("float32")
    return dtype
