"""Partial autocorrelation feature written against backend-agnostic ops."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict, cast

from backend_agnostic.ops import ArrayT, Ops, backend_ops


class ArrayFeature(Protocol):
    """A callable that maps an array to same-backend output, for whichever backend
    is active at call time. `ArrayT` is scoped to `__call__`, not to the class:
    nothing about an `ArrayFeature` instance itself is tied to one backend."""

    def __call__(self, x: ArrayT) -> ArrayT: ...

    @property
    def names(self) -> list[str]: ...


class PartialAutocorrelationParams(TypedDict):
    """Parameters for one partial autocorrelation output column.

    Attributes:
        lag: Lag index to evaluate. ``0`` returns 1; values beyond the computable
            recursion range return 0.
    """

    lag: int


class PartialAutocorrelation(ArrayFeature):
    r"""Partial autocorrelation using the globally configured backend.

    The public API mirrors tstorch's feature class shape: construct with a list of
    parameter dicts, call with ``(B, S, T)`` data, and receive ``(B, S, F)`` output.
    The calculation itself is written once against a backend ops namespace.

    Args:
        params: One dict per requested output, with a ``lag`` key.
    """

    def __init__(self, params: Sequence[PartialAutocorrelationParams]) -> None:
        self.params: list[PartialAutocorrelationParams] = list(params)
        self._lags: tuple[int, ...] = tuple(p["lag"] for p in self.params)
        self._global_maxlag: int = max(self._lags, default=0)

    def __call__(self, x: ArrayT) -> ArrayT:
        # A subscripted `cast(Ops[ArrayT], ...)` would evaluate `Ops[ArrayT]` at
        # runtime into a `_GenericAlias` object, which `torch.compile(fullgraph=True)`
        # cannot trace through. The string form is never evaluated at runtime.
        ops = cast("Ops[ArrayT]", backend_ops())
        return partial_autocorrelation(x, self._lags, self._global_maxlag, ops)

    @property
    def names(self) -> list[str]:
        """Output names in the same order as ``params``."""
        return [f"partial_autocorrelation__lag_{lag}" for lag in self._lags]


def partial_autocorrelation(
    x: ArrayT, lags: tuple[int, ...], global_maxlag: int, ops: Ops[ArrayT]
) -> ArrayT:
    batch_size, n_states, T = ops.shape(x)
    K = len(lags)
    if K == 0:
        return ops.zeros((batch_size, n_states, 0), x)

    maxlag = min(global_maxlag, T - 1)
    if maxlag < 1:
        columns = [
            ops.ones((batch_size, n_states), x)
            if lag == 0
            else ops.zeros((batch_size, n_states), x)
            for lag in lags
        ]
        return ops.stack(columns, axis=-1)

    x_work = ops.work_array(x)
    acf = adjusted_acf(x_work, maxlag, ops)
    phi = ops.zeros((maxlag + 1, batch_size, n_states), x_work)
    pacf = ops.zeros((maxlag + 1, batch_size, n_states), x_work)
    pacf = ops.set_axis0(pacf, 0, ops.ones((batch_size, n_states), x_work))
    phi = ops.set_axis0(phi, 1, acf[1])
    pacf = ops.set_axis0(pacf, 1, acf[1])

    for k in range(2, maxlag + 1):
        acf_prev = acf[1:k]
        phi_prev = phi[1:k]
        num = acf[k] - ops.sum(phi_prev * ops.flip(acf_prev, axis=0), axis=0)
        denom = 1.0 - ops.sum(phi_prev * acf_prev, axis=0)
        phi_k = num / denom
        pacf = ops.set_axis0(pacf, k, phi_k)
        updated_phi = phi_prev - ops.expand_dims(phi_k, axis=0) * ops.flip(phi_prev, axis=0)
        phi = ops.set_axis0_slice(phi, 1, k, updated_phi)
        phi = ops.set_axis0(phi, k, phi_k)

    pacf_bsk = ops.transpose(pacf, (1, 2, 0))
    zero_column = ops.zeros_like(pacf_bsk[..., 0])
    columns = [pacf_bsk[..., lag] if lag <= maxlag else zero_column for lag in lags]
    return ops.astype(ops.stack(columns, axis=-1), ops.dtype(x))


def adjusted_acf(x: ArrayT, maxlag: int, ops: Ops[ArrayT]) -> ArrayT:
    batch_size, n_states, T = ops.shape(x)
    x_centered = x - ops.mean(x, axis=2, keepdims=True)
    var0 = ops.sum(x_centered**2, axis=2)
    acf_values = [ops.ones((batch_size, n_states), x)]
    for lag in range(1, maxlag + 1):
        raw = ops.sum(x_centered[..., :-lag] * x_centered[..., lag:], axis=2)
        acf_values.append(raw * T / ((T - lag) * var0))
    return ops.stack(acf_values, axis=0)
