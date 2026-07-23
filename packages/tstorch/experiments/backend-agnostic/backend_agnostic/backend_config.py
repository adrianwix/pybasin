# pyright: basic
"""Global backend configuration for the backend-agnostic feature experiment."""

from __future__ import annotations

from typing import Literal

BackendName = Literal["numpy", "torch", "jax"]
_BACKEND: BackendName = "numpy"


def backend() -> BackendName:
    """Return the globally configured feature backend."""
    return _BACKEND


def set_backend(name: BackendName) -> None:
    """Set the global backend used by backend-agnostic feature kernels."""
    global _BACKEND
    if name not in {"numpy", "torch", "jax"}:
        raise ValueError(f"unknown backend: {name}")
    _BACKEND = name
