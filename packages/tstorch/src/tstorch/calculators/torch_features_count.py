"""Counting feature calculators."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor

from tstorch.feature_factory import Feature, ParameterlessFeature


class CountAboveParams(TypedDict):
    """Parameters for one `CountAbove` output column.

    Attributes:
        t: Inclusive lower threshold.
    """

    t: float


_COUNT_ABOVE_DEFAULT_PARAMS: list[CountAboveParams] = [{"t": 0.0}]


class CountAbove(Feature[CountAboveParams]):
    r"""Fraction of values greater than or equal to each threshold.

    For a series of length \(T\) and threshold \(t\), the feature returns

    \[
    \frac{1}{T}\sum_{i=1}^{T}\mathbf{1}[x_i \ge t].
    \]

    Several thresholds are evaluated together by broadcasting one parameter axis.

    Args:
        params: One dict per requested output, each with a ``t`` key. Defaults to
            tsfresh's comprehensive setting, ``[{"t": 0.0}]``, when omitted. An
            explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[CountAboveParams] | None = None) -> None:
        if params is None:
            params = list(_COUNT_ABOVE_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "CountAbove requires at least one param; omit `params` to use the "
                f"default {_COUNT_ABOVE_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[CountAboveParams] = params
        self._thresholds = torch.tensor([param["t"] for param in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Independent thresholds broadcast through one comparison without a Python param loop.
        thresholds = self._thresholds.to(dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        counts = (x.unsqueeze(-1) >= thresholds).sum(dim=2)
        return counts.to(x.dtype) / x.shape[2]

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format: ``count_above__t_<t>``.
        """
        return [f"count_above__t_{param['t']}" for param in self.params]


class CountAboveMean(ParameterlessFeature):
    """Number of values strictly greater than the series mean.

    Values equal to the mean are not counted. If the mean is NaN, every comparison is
    false and the result is zero, matching tsfresh.

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
        mean = x.mean(dim=2, keepdim=True)
        return (x > mean).sum(dim=2, keepdim=True).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``count_above_mean``.
        """
        return ["count_above_mean"]


class CountBelowParams(TypedDict):
    """Parameters for one `CountBelow` output column.

    Attributes:
        t: Inclusive upper threshold.
    """

    t: float


_COUNT_BELOW_DEFAULT_PARAMS: list[CountBelowParams] = [{"t": 0.0}]


class CountBelow(Feature[CountBelowParams]):
    r"""Fraction of values less than or equal to each threshold.

    For a series of length \(T\) and threshold \(t\), the feature returns

    \[
    \frac{1}{T}\sum_{i=1}^{T}\mathbf{1}[x_i \le t].
    \]

    Several thresholds are evaluated together by broadcasting one parameter axis.

    Args:
        params: One dict per requested output, each with a ``t`` key. Defaults to
            tsfresh's comprehensive setting, ``[{"t": 0.0}]``, when omitted. An
            explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[CountBelowParams] | None = None) -> None:
        if params is None:
            params = list(_COUNT_BELOW_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "CountBelow requires at least one param; omit `params` to use the "
                f"default {_COUNT_BELOW_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[CountBelowParams] = params
        self._thresholds = torch.tensor([param["t"] for param in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Independent thresholds broadcast through one comparison without a Python param loop.
        thresholds = self._thresholds.to(dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        counts = (x.unsqueeze(-1) <= thresholds).sum(dim=2)
        return counts.to(x.dtype) / x.shape[2]

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format: ``count_below__t_<t>``.
        """
        return [f"count_below__t_{param['t']}" for param in self.params]


class CountBelowMean(ParameterlessFeature):
    """Number of values strictly less than the series mean.

    Values equal to the mean are not counted. If the mean is NaN, every comparison is
    false and the result is zero, matching tsfresh.

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
        mean = x.mean(dim=2, keepdim=True)
        return (x < mean).sum(dim=2, keepdim=True).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``count_below_mean``.
        """
        return ["count_below_mean"]


class CountInRangeParams(TypedDict):
    """Parameters for one `CountInRange` output column.

    Attributes:
        min_val: Inclusive lower bound.
        max_val: Exclusive upper bound.
    """

    min_val: float
    max_val: float


_COUNT_IN_RANGE_DEFAULT_PARAMS: list[CountInRangeParams] = [
    {"min_val": -1.0, "max_val": 1.0},
    {"min_val": -1e12, "max_val": 0.0},
    {"min_val": 0.0, "max_val": 1e12},
]


class CountInRange(Feature[CountInRangeParams]):
    r"""Count values within each half-open interval.

    For lower and upper bounds \(a\) and \(b\), the result is

    \[
    \sum_{i=1}^{T}\mathbf{1}[a \le x_i < b].
    \]

    The lower bound is inclusive and the upper bound is exclusive. Multiple intervals
    are evaluated together by broadcasting one parameter axis.

    Args:
        params: One dict per requested output, each with ``min_val`` and ``max_val``
            keys. Defaults to tsfresh's three comprehensive ``range_count`` intervals
            when omitted. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[CountInRangeParams] | None = None) -> None:
        if params is None:
            params = list(_COUNT_IN_RANGE_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "CountInRange requires at least one param; omit `params` to use the "
                f"default {_COUNT_IN_RANGE_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[CountInRangeParams] = params
        self._minimums = torch.tensor([param["min_val"] for param in params], dtype=torch.float64)
        self._maximums = torch.tensor([param["max_val"] for param in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Independent ranges broadcast through one pair of comparisons without a param loop.
        minimums = self._minimums.to(dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        maximums = self._maximums.to(dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        values = x.unsqueeze(-1)
        return ((values >= minimums) & (values < maximums)).sum(dim=2).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format:
            ``count_in_range__min_val_<min_val>__max_val_<max_val>``.
        """
        return [
            (f"count_in_range__min_val_{param['min_val']}__max_val_{param['max_val']}")
            for param in self.params
        ]


class CountValueParams(TypedDict):
    """Parameters for one `CountValue` output column.

    Attributes:
        value: Value whose occurrences are counted. NaN selects NaN entries.
    """

    value: float


_COUNT_VALUE_DEFAULT_PARAMS: list[CountValueParams] = [
    {"value": 0.0},
    {"value": 1.0},
    {"value": -1.0},
]


class CountValue(Feature[CountValueParams]):
    r"""Count occurrences of each requested value.

    For a finite value \(v\), the result is

    \[
    \sum_{i=1}^{T}\mathbf{1}[x_i = v].
    \]

    A requested NaN value counts NaN entries explicitly, matching tsfresh. Several
    values are evaluated together by broadcasting one parameter axis.

    Args:
        params: One dict per requested output, each with a ``value`` key. Defaults to
            tsfresh's comprehensive values ``0``, ``1``, and ``-1`` when omitted. An
            explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[CountValueParams] | None = None) -> None:
        if params is None:
            params = list(_COUNT_VALUE_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "CountValue requires at least one param; omit `params` to use the "
                f"default {_COUNT_VALUE_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[CountValueParams] = params
        self._values = torch.tensor([param["value"] for param in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Independent target values broadcast through one comparison without a Python param loop.
        values = self._values.to(dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        samples = x.unsqueeze(-1)
        matches = torch.where(torch.isnan(values), torch.isnan(samples), samples == values)
        return matches.sum(dim=2).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format: ``count_value__value_<value>``.
        """
        return [f"count_value__value_{param['value']}" for param in self.params]
