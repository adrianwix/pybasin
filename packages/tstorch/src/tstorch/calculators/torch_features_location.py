"""Location feature calculators."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor

from tstorch.feature_factory import Feature, ParameterlessFeature


class FirstLocationOfMaximum(ParameterlessFeature):
    r"""Relative first location of the maximum value.

    For a series of length \(T\), this returns \(i / T\), where \(i\) is the
    zero-based index of the first maximum. Ties therefore resolve to the earliest
    maximum. An empty series returns NaN.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1).
        """
        time_steps = x.shape[2]
        if time_steps == 0:
            return torch.full((*x.shape[:2], 1), float("nan"), dtype=x.dtype, device=x.device)
        index = x.argmax(dim=2, keepdim=True)
        return index.to(x.dtype) / time_steps

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``first_location_of_maximum``.
        """
        return ["first_location_of_maximum"]


class FirstLocationOfMinimum(ParameterlessFeature):
    r"""Relative first location of the minimum value.

    For a series of length \(T\), this returns \(i / T\), where \(i\) is the
    zero-based index of the first minimum. Ties therefore resolve to the earliest
    minimum. An empty series returns NaN.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1).
        """
        time_steps = x.shape[2]
        if time_steps == 0:
            return torch.full((*x.shape[:2], 1), float("nan"), dtype=x.dtype, device=x.device)
        index = x.argmin(dim=2, keepdim=True)
        return index.to(x.dtype) / time_steps

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``first_location_of_minimum``.
        """
        return ["first_location_of_minimum"]


class LastLocationOfMaximum(ParameterlessFeature):
    r"""Relative last location of the maximum value.

    For a series of length \(T\), this returns \(1 - j / T\), where \(j\) is the
    zero-based index of the first maximum in the reversed series. This is
    \((i + 1) / T\) for the zero-based index \(i\) of the last maximum in the
    original series. An empty series returns NaN.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1).
        """
        time_steps = x.shape[2]
        if time_steps == 0:
            return torch.full((*x.shape[:2], 1), float("nan"), dtype=x.dtype, device=x.device)
        index_from_end = x.flip(dims=(2,)).argmax(dim=2, keepdim=True)
        return 1.0 - index_from_end.to(x.dtype) / time_steps

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``last_location_of_maximum``.
        """
        return ["last_location_of_maximum"]


class LastLocationOfMinimum(ParameterlessFeature):
    r"""Relative last location of the minimum value.

    For a series of length \(T\), this returns \(1 - j / T\), where \(j\) is the
    zero-based index of the first minimum in the reversed series. This is
    \((i + 1) / T\) for the zero-based index \(i\) of the last minimum in the
    original series. An empty series returns NaN.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1).
        """
        time_steps = x.shape[2]
        if time_steps == 0:
            return torch.full((*x.shape[:2], 1), float("nan"), dtype=x.dtype, device=x.device)
        index_from_end = x.flip(dims=(2,)).argmin(dim=2, keepdim=True)
        return 1.0 - index_from_end.to(x.dtype) / time_steps

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``last_location_of_minimum``.
        """
        return ["last_location_of_minimum"]


class IndexMassQuantileParams(TypedDict):
    """Parameters for one `IndexMassQuantile` output column.

    Attributes:
        q: Fraction of cumulative absolute mass to locate.
    """

    q: float


_INDEX_MASS_QUANTILE_DEFAULT_PARAMS: list[IndexMassQuantileParams] = [
    {"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
]


class IndexMassQuantile(Feature[IndexMassQuantileParams]):
    r"""Relative index where each requested fraction of absolute mass is reached.

    For \(q\), this computes the first index \(i\) satisfying

    \[
    \sum_{t=1}^{i} |x_t| \ge q \sum_{t=1}^{T} |x_t|
    \]

    and returns \(i / T\), using one-based \(i\). All-zero and empty series return
    NaN. For a nonzero series, \(q=1\) always returns the final position. Several
    quantiles share one cumulative-sum pass and broadcast their thresholds across a
    parameter axis.

    Args:
        params: One dict per requested output, each with a ``q`` key. Defaults to
            tsfresh's comprehensive setting of eight quantiles from 0.1 through 0.9.
            An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[IndexMassQuantileParams] | None = None) -> None:
        if params is None:
            params = list(_INDEX_MASS_QUANTILE_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "IndexMassQuantile requires at least one param; omit `params` to use the "
                f"default {_INDEX_MASS_QUANTILE_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[IndexMassQuantileParams] = params
        self._quantiles: Tensor = torch.tensor(
            [param["q"] for param in params], dtype=torch.float64
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        time_steps = x.shape[2]
        if time_steps == 0:
            return torch.full(
                (*x.shape[:2], len(self.params)),
                float("nan"),
                dtype=x.dtype,
                device=x.device,
            )

        # Every q reads a threshold crossing from one shared cumulative-mass pass.
        cumulative_mass = x.abs().cumsum(dim=2)
        total_mass = cumulative_mass[:, :, -1:]
        quantiles = self._quantiles.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
        thresholds = total_mass * quantiles
        indices = torch.searchsorted(cumulative_mass, thresholds)
        indices = torch.where(indices == time_steps, 0, indices)
        result = (indices.to(x.dtype) + 1.0) / time_steps
        return torch.where(
            total_mass == 0,
            torch.full_like(result, float("nan")),
            result,
        )

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format: ``index_mass_quantile__q_<q>``.
        """
        return [f"index_mass_quantile__q_{param['q']}" for param in self.params]
