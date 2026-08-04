# pyright: basic
"""Public facade for the backend-agnostic feature experiment."""

from __future__ import annotations

from backend_agnostic.backend_config import BackendName, backend, set_backend
from backend_agnostic.partial_autocorrelation import (
    PartialAutocorrelation,
    PartialAutocorrelationParams,
    adjusted_acf,
    partial_autocorrelation,
)

__all__ = [
    "BackendName",
    "PartialAutocorrelation",
    "PartialAutocorrelationParams",
    "adjusted_acf",
    "backend",
    "partial_autocorrelation",
    "set_backend",
]
