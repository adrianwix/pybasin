"""Shared CPU/GPU array-backend resolution.

Every pybasin component that has a CPU (numpy) and GPU (cupy) code path --
clustering/classification predictors, feature selectors -- resolves its
backend through :func:`resolve_array_backend` and crosses the host/device
boundary only through :func:`to_native`/:func:`from_native`/:func:`to_numpy`.
This module owns device resolution and array conversion only; it knows
nothing about any particular estimator library. Domain-specific modules
(e.g. ``pybasin.predictors._device_backend``) build their own richer backend
type on top of :class:`ArrayBackend` when they need estimator classes or
metric functions alongside it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, Protocol, TypeVar, overload, runtime_checkable

import numpy as np
import torch

BackendName = Literal["cpu", "gpu"]

Array = np.ndarray | torch.Tensor
"""The two array/tensor types every public pybasin signature accepts -- a named
alias so it's written once instead of repeated at every call site."""

TArray = TypeVar("TArray", bound=Array)
"""Binds a single call's input array/tensor type to its mirrored return type.

Used by functions and methods whose contract is "the output has the same
concrete type (``np.ndarray`` or ``torch.Tensor``) as this particular input" --
e.g. :func:`from_native`, or ``CorrelationSelector.transform``. A plain ``Array``
union on both sides would allow numpy-in/tensor-out as a valid typing, silently
dropping the actual invariant; binding both occurrences to the same ``TArray``
makes the type checker enforce it.
"""


@runtime_checkable
class SupportsArrayBackend(Protocol):
    """Structural type for anything with a resolved array module and device family.

    :func:`to_native`/:func:`to_numpy`/:func:`from_native` only ever read ``is_gpu``
    and ``xp`` off the backend they're given, so any domain-specific backend type
    (e.g. ``pybasin.predictors._device_backend.DeviceBackend``, which carries extra
    estimator-class fields alongside these two) satisfies this protocol for free.
    Declared as properties (read-only) so frozen dataclasses -- whose fields are
    themselves read-only -- satisfy it structurally.
    """

    @property
    def is_gpu(self) -> bool: ...

    @property
    def xp(self) -> ModuleType: ...


@dataclass(frozen=True)
class ArrayBackend:
    """A resolved array backend: which device family, and its array module.

    :ivar name: ``"cpu"`` or ``"gpu"``.
    :ivar is_gpu: Whether this backend runs on GPU (cupy).
    :ivar fallback_reason: Why a fallback occurred, or ``None`` if the resolved
        backend is exactly what was requested.
    :ivar xp: ``numpy`` or ``cupy``.
    """

    name: BackendName
    is_gpu: bool
    fallback_reason: str | None
    xp: ModuleType


def device_kind(X: Any) -> str:
    """Return 'cuda', 'cpu', or another torch device-type string for a tensor/array."""
    if isinstance(X, torch.Tensor):
        return X.device.type
    if type(X).__module__.startswith("cupy"):
        return "cuda"
    return "cpu"


def _cpu_array_backend(fallback_reason: str | None = None) -> ArrayBackend:
    return ArrayBackend(name="cpu", is_gpu=False, fallback_reason=fallback_reason, xp=np)


def _gpu_array_backend() -> ArrayBackend:
    try:
        import cupy
    except ImportError as e:
        raise ImportError(
            "GPU backend requested but cupy is not installed. "
            "Install RAPIDS cuML (which bundles cupy), see https://docs.rapids.ai/install"
        ) from e

    return ArrayBackend(name="gpu", is_gpu=True, fallback_reason=None, xp=cupy)


def resolve_array_backend(
    X: Any,
    *,
    requested: Literal["auto", "cpu", "gpu"] = "auto",
    fallback: Literal["raise", "cpu"] = "raise",
) -> ArrayBackend:
    """Resolve which backend (CPU/numpy or GPU/cupy) a call against ``X`` should use.

    ``requested="auto"`` (the default) infers the backend from ``X``'s device:
    a CPU tensor or plain ``numpy``/``cupy`` array resolves accordingly; a CUDA
    tensor resolves to GPU, raising ``ImportError`` if cupy isn't installed
    unless ``fallback="cpu"``. Any other device (``mps``, ``xpu``, ...) always
    warns and falls back to CPU, regardless of ``fallback``.

    ``requested="cpu"``/``"gpu"`` force that backend; an explicit ``"gpu"``
    request that cupy can't satisfy always raises, regardless of ``fallback``.
    """
    if requested == "cpu":
        return _cpu_array_backend()
    if requested == "gpu":
        return _gpu_array_backend()

    kind = device_kind(X)
    if kind == "cuda":
        try:
            return _gpu_array_backend()
        except ImportError:
            if fallback == "cpu":
                return _cpu_array_backend(fallback_reason="cupy not installed; fell back to CPU")
            raise
    if kind == "cpu":
        return _cpu_array_backend()

    warnings.warn(
        f"No GPU backend for device {kind!r}; falling back to CPU.",
        stacklevel=2,
    )
    return _cpu_array_backend(fallback_reason=f"unsupported device {kind!r}; fell back to CPU")


def to_native(X: Any, *, backend: SupportsArrayBackend) -> Any:
    """Convert ``X`` to the array type ``backend`` operates on (numpy or cupy)."""
    if backend.is_gpu:
        cp = backend.xp
        if isinstance(X, torch.Tensor):
            if X.device.type == "cuda":
                return cp.from_dlpack(X)
            return cp.asarray(X.detach().cpu().numpy())
        if type(X).__module__.startswith("cupy"):
            return X
        return cp.asarray(X)

    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if type(X).__module__.startswith("cupy"):
        return backend.xp.asnumpy(X)
    return X


def to_numpy(x: Any, *, backend: SupportsArrayBackend) -> np.ndarray:
    """Unconditionally land ``x`` on host as a ``numpy`` array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if type(x).__module__.startswith("cupy"):
        return backend.xp.asnumpy(x)
    return np.asarray(x)


@overload
def from_native(
    labels: Any, *, like: torch.Tensor, backend: SupportsArrayBackend
) -> torch.Tensor: ...
@overload
def from_native(labels: Any, *, like: np.ndarray, backend: SupportsArrayBackend) -> np.ndarray: ...
def from_native(labels: Any, *, like: Any, backend: SupportsArrayBackend) -> Any:
    """Convert ``labels`` (produced by ``backend``) to mirror ``like``'s type/device."""
    if isinstance(like, torch.Tensor):
        if type(labels).__module__.startswith("cupy"):
            native_tensor = torch.from_dlpack(labels)
        else:
            native_tensor = torch.from_numpy(np.asarray(labels))  # type: ignore[arg-type]
        return native_tensor.to(device=like.device)

    if type(like).__module__.startswith("cupy"):
        if type(labels).__module__.startswith("cupy"):
            return labels
        return backend.xp.asarray(labels)

    return to_numpy(labels, backend=backend)
