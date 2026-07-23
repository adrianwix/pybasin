# pyright: basic
"""Backend primitive operations for the backend-agnostic feature experiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import torch
from backend_config import BackendName, backend
from torch import Tensor


class JaxArray(Protocol):
    shape: tuple[int, ...]
    dtype: object
    at: object

    def astype(self, dtype: object) -> JaxArray: ...

    def mean(self, axis: int, keepdims: bool) -> JaxArray: ...

    def sum(self, axis: int) -> JaxArray: ...


type BackendArray = Tensor | np.ndarray | JaxArray
type DType = torch.dtype | np.dtype | object


class Ops(Protocol):
    name: BackendName

    def shape(self, x: BackendArray) -> tuple[int, ...]: ...

    def dtype(self, x: BackendArray) -> DType: ...

    def work_array(self, x: BackendArray) -> BackendArray: ...

    def zeros(self, shape: tuple[int, ...], like: BackendArray) -> BackendArray: ...

    def ones(self, shape: tuple[int, ...], like: BackendArray) -> BackendArray: ...

    def zeros_like(self, x: BackendArray) -> BackendArray: ...

    def mean(self, x: BackendArray, axis: int, keepdims: bool) -> BackendArray: ...

    def sum(self, x: BackendArray, axis: int) -> BackendArray: ...

    def flip(self, x: BackendArray, axis: int) -> BackendArray: ...

    def stack(self, xs: Sequence[BackendArray], axis: int) -> BackendArray: ...

    def transpose(self, x: BackendArray, axes: tuple[int, ...]) -> BackendArray: ...

    def expand_dims(self, x: BackendArray, axis: int) -> BackendArray: ...

    def astype(self, x: BackendArray, dtype: DType) -> BackendArray: ...

    def set_axis0(self, x: BackendArray, index: int, value: BackendArray) -> BackendArray: ...

    def set_axis0_slice(
        self, x: BackendArray, start: int, stop: int, value: BackendArray
    ) -> BackendArray: ...


class TorchOps:
    name: BackendName = "torch"

    def shape(self, x: Tensor) -> tuple[int, ...]:
        return tuple(x.shape)

    def dtype(self, x: Tensor) -> torch.dtype:
        return x.dtype

    def work_array(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"torch backend expects torch.Tensor input, got {type(x)!r}")
        return x.to(torch.float64)

    def zeros(self, shape: tuple[int, ...], like: Tensor) -> Tensor:
        return torch.zeros(shape, dtype=like.dtype, device=like.device)

    def ones(self, shape: tuple[int, ...], like: Tensor) -> Tensor:
        return torch.ones(shape, dtype=like.dtype, device=like.device)

    def zeros_like(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)

    def mean(self, x: Tensor, axis: int, keepdims: bool) -> Tensor:
        return x.mean(dim=axis, keepdim=keepdims)

    def sum(self, x: Tensor, axis: int) -> Tensor:
        return x.sum(dim=axis)

    def flip(self, x: Tensor, axis: int) -> Tensor:
        return x.flip(dims=[axis])

    def stack(self, xs: Sequence[Tensor], axis: int) -> Tensor:
        return torch.stack(tuple(xs), dim=axis)

    def transpose(self, x: Tensor, axes: tuple[int, ...]) -> Tensor:
        return x.permute(*axes)

    def expand_dims(self, x: Tensor, axis: int) -> Tensor:
        return x.unsqueeze(axis)

    def astype(self, x: Tensor, dtype: torch.dtype) -> Tensor:
        return x.to(dtype)

    def set_axis0(self, x: Tensor, index: int, value: Tensor) -> Tensor:
        result = x.clone()
        result[index] = value
        return result

    def set_axis0_slice(self, x: Tensor, start: int, stop: int, value: Tensor) -> Tensor:
        result = x.clone()
        result[start:stop] = value
        return result


class NumpyOps:
    name: BackendName = "numpy"

    def shape(self, x: np.ndarray) -> tuple[int, ...]:
        return tuple(x.shape)

    def dtype(self, x: np.ndarray) -> np.dtype:
        return x.dtype

    def work_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise TypeError(f"numpy backend expects numpy.ndarray input, got {type(x)!r}")
        return x.astype(np.float64)

    def zeros(self, shape: tuple[int, ...], like: np.ndarray) -> np.ndarray:
        return np.zeros(shape, dtype=like.dtype)

    def ones(self, shape: tuple[int, ...], like: np.ndarray) -> np.ndarray:
        return np.ones(shape, dtype=like.dtype)

    def zeros_like(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def mean(self, x: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
        return x.mean(axis=axis, keepdims=keepdims)

    def sum(self, x: np.ndarray, axis: int) -> np.ndarray:
        return x.sum(axis=axis)

    def flip(self, x: np.ndarray, axis: int) -> np.ndarray:
        return np.flip(x, axis=axis)

    def stack(self, xs: Sequence[np.ndarray], axis: int) -> np.ndarray:
        return np.stack(xs, axis=axis)

    def transpose(self, x: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        return np.transpose(x, axes)

    def expand_dims(self, x: np.ndarray, axis: int) -> np.ndarray:
        return np.expand_dims(x, axis=axis)

    def astype(self, x: np.ndarray, dtype: np.dtype) -> np.ndarray:
        return x.astype(dtype)

    def set_axis0(self, x: np.ndarray, index: int, value: np.ndarray) -> np.ndarray:
        result = x.copy()
        result[index] = value
        return result

    def set_axis0_slice(
        self, x: np.ndarray, start: int, stop: int, value: np.ndarray
    ) -> np.ndarray:
        result = x.copy()
        result[start:stop] = value
        return result


class JaxOps:
    name: BackendName = "jax"

    def shape(self, x: JaxArray) -> tuple[int, ...]:
        return tuple(x.shape)

    def dtype(self, x: JaxArray) -> object:
        return x.dtype

    def work_array(self, x: JaxArray) -> JaxArray:
        if type(x).__module__.split(".", maxsplit=1)[0] != "jax":
            raise TypeError(f"jax backend expects jax array input, got {type(x)!r}")
        jnp = _jax_numpy()
        return x.astype(jnp.result_type(x, jnp.float32))

    def zeros(self, shape: tuple[int, ...], like: JaxArray) -> JaxArray:
        jnp = _jax_numpy()
        return jnp.zeros(shape, dtype=like.dtype)

    def ones(self, shape: tuple[int, ...], like: JaxArray) -> JaxArray:
        jnp = _jax_numpy()
        return jnp.ones(shape, dtype=like.dtype)

    def zeros_like(self, x: JaxArray) -> JaxArray:
        jnp = _jax_numpy()
        return jnp.zeros_like(x)

    def mean(self, x: JaxArray, axis: int, keepdims: bool) -> JaxArray:
        return x.mean(axis=axis, keepdims=keepdims)

    def sum(self, x: JaxArray, axis: int) -> JaxArray:
        return x.sum(axis=axis)

    def flip(self, x: JaxArray, axis: int) -> JaxArray:
        jnp = _jax_numpy()
        return jnp.flip(x, axis=axis)

    def stack(self, xs: Sequence[JaxArray], axis: int) -> JaxArray:
        jnp = _jax_numpy()
        return jnp.stack(xs, axis=axis)

    def transpose(self, x: JaxArray, axes: tuple[int, ...]) -> JaxArray:
        jnp = _jax_numpy()
        return jnp.transpose(x, axes)

    def expand_dims(self, x: JaxArray, axis: int) -> JaxArray:
        jnp = _jax_numpy()
        return jnp.expand_dims(x, axis=axis)

    def astype(self, x: JaxArray, dtype: object) -> JaxArray:
        return x.astype(dtype)

    def set_axis0(self, x: JaxArray, index: int, value: JaxArray) -> JaxArray:
        return x.at[index].set(value)

    def set_axis0_slice(self, x: JaxArray, start: int, stop: int, value: JaxArray) -> JaxArray:
        return x.at[start:stop].set(value)


_OPS: dict[BackendName, Ops] = {
    "numpy": NumpyOps(),
    "torch": TorchOps(),
    "jax": JaxOps(),
}


def backend_ops() -> Ops:
    """Resolve the ops namespace for the globally configured backend."""
    return _OPS[backend()]


def _jax_numpy() -> Any:
    import jax.numpy as jnp  # noqa: PLC0415

    return jnp
