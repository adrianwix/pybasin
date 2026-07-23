from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import torch
from torch import Tensor

from tstorch.feature_factory import Feature
from tstorch.scipy.stats import linregress
from tstorch.statsmodels.tsa import auto_reg
from tstorch.statsmodels.tsa.stattools import Autolag, adfuller

TrendAttribute = Literal["slope", "intercept", "rvalue", "pvalue", "stderr"]
Aggregate = Literal["mean", "var", "min", "max"]
AdfAttribute = Literal["teststat", "pvalue", "usedlag"]

_TREND_ATTR_INDEX: dict[TrendAttribute, int] = {
    "slope": 0,
    "intercept": 1,
    "rvalue": 2,
    "pvalue": 3,
    "stderr": 4,
}
_AGGREGATE_INDEX: dict[Aggregate, int] = {
    "mean": 0,
    "var": 1,
    "min": 2,
    "max": 3,
}
_ADF_ATTR_INDEX: dict[AdfAttribute, int] = {
    "teststat": 0,
    "pvalue": 1,
    "usedlag": 2,
}


class LinearTrendParams(TypedDict):
    """Parameters for one `LinearTrend` output column.

    Attributes:
        attr: Regression statistic to return.
    """

    attr: TrendAttribute


class LinearTrendTimewiseParams(TypedDict):
    """Parameters for one `LinearTrendTimewise` output column.

    Attributes:
        attr: Regression statistic to return.
    """

    attr: TrendAttribute


class AggLinearTrendParams(TypedDict):
    """Parameters for one `AggLinearTrend` output column.

    Attributes:
        chunk_size: Number of consecutive samples in each chunk.
        f_agg: Reduction applied to each chunk.
        attr: Regression statistic to return.
    """

    chunk_size: int
    f_agg: Aggregate
    attr: TrendAttribute


class ArCoefficientParams(TypedDict):
    """Parameters for one `ArCoefficient` output column.

    Attributes:
        k: Maximum autoregressive lag.
        coeff: Coefficient index, with zero denoting the intercept.
    """

    k: int
    coeff: int


class AugmentedDickeyFullerParams(TypedDict):
    """Parameters for one `AugmentedDickeyFuller` output column.

    Attributes:
        attr: Test result to return.
        autolag: Lag-selection method. Omission is equivalent to ``"AIC"``.
    """

    attr: AdfAttribute
    autolag: NotRequired[Autolag]


_LINEAR_TREND_DEFAULT_PARAMS: list[LinearTrendParams] = [
    {"attr": "pvalue"},
    {"attr": "rvalue"},
    {"attr": "intercept"},
    {"attr": "slope"},
    {"attr": "stderr"},
]
_LINEAR_TREND_TIMEWISE_DEFAULT_PARAMS: list[LinearTrendTimewiseParams] = [
    {"attr": "pvalue"},
    {"attr": "rvalue"},
    {"attr": "intercept"},
    {"attr": "slope"},
    {"attr": "stderr"},
]
_AGG_LINEAR_TREND_DEFAULT_PARAMS: list[AggLinearTrendParams] = [
    {"attr": attr, "chunk_size": chunk_size, "f_agg": f_agg}
    for attr in ("rvalue", "intercept", "slope", "stderr")
    for chunk_size in (5, 10, 50)
    for f_agg in ("max", "min", "mean", "var")
]
_AR_COEFFICIENT_DEFAULT_PARAMS: list[ArCoefficientParams] = [
    {"coeff": coeff, "k": 10} for coeff in range(11)
]
_ADF_DEFAULT_PARAMS: list[AugmentedDickeyFullerParams] = [
    {"attr": "teststat"},
    {"attr": "pvalue"},
    {"attr": "usedlag"},
]


def _select_columns(stats: Tensor, indices: Tensor) -> Tensor:
    """Select the requested final-axis columns from a shared statistic tensor."""
    output_count = indices.shape[0]
    index = indices.to(device=stats.device).view(*([1] * (stats.ndim - 1)), output_count)
    return stats.gather(dim=-1, index=index.expand(*stats.shape[:-1], output_count))


class LinearTrend(Feature[LinearTrendParams]):
    r"""Ordinary least-squares trend statistics against the sample index.

    The fitted model is \(x_t = \beta_1 t + \beta_0 + \varepsilon_t\), with
    \(t = 0, \ldots, T-1\). The reported values match SciPy's ``linregress`` statistics.
    A constant series has an undefined correlation, p-value, and standard error.

    Args:
        params: One dict per requested statistic. Defaults to all five statistics when
            omitted. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[LinearTrendParams] | None = None) -> None:
        if params is None:
            params = list(_LINEAR_TREND_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "LinearTrend requires at least one param; omit `params` to use the "
                f"default {_LINEAR_TREND_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[LinearTrendParams] = params
        self._attr_indices = torch.tensor(
            [_TREND_ATTR_INDEX[param["attr"]] for param in params],
            dtype=torch.long,
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # All requested attributes reuse one regression.
        return _select_columns(linregress(x), self._attr_indices)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``linear_trend__attr_<attr>``.
        """
        return [f"linear_trend__attr_{param['attr']}" for param in self.params]


class LinearTrendTimewise(Feature[LinearTrendTimewiseParams]):
    r"""Ordinary least-squares trend statistics against elapsed time in hours.

    Samples are treated as one second apart, matching the timestamp index supplied by tstorch's
    tsfresh interoperability layer. Thus the regression axis is
    \(0, 1/3600, \ldots, (T-1)/3600\) hours. Scale-dependent slope and standard-error values
    are 3600 times their sample-index counterparts; the other statistics are unchanged.

    Args:
        params: One dict per requested statistic. Defaults to all five statistics when
            omitted. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[LinearTrendTimewiseParams] | None = None) -> None:
        if params is None:
            params = list(_LINEAR_TREND_TIMEWISE_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "LinearTrendTimewise requires at least one param; omit `params` to use the "
                f"default {_LINEAR_TREND_TIMEWISE_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[LinearTrendTimewiseParams] = params
        self._attr_indices = torch.tensor(
            [_TREND_ATTR_INDEX[param["attr"]] for param in params],
            dtype=torch.long,
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T), sampled once per second.

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # A uniform rescaling of time only changes slope and its standard error.
        stats = linregress(x)
        scale = torch.tensor(
            [3600.0, 1.0, 1.0, 1.0, 3600.0],
            dtype=x.dtype,
            device=x.device,
        )
        return _select_columns(stats * scale, self._attr_indices)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``linear_trend_timewise__attr_<attr>``.
        """
        return [f"linear_trend_timewise__attr_{param['attr']}" for param in self.params]


def _aggregate_chunks(x: Tensor, chunk_size: int) -> Tensor:
    """Return mean, variance, minimum, and maximum for each chunk."""
    batch_size, n_states, time_steps = x.shape
    chunk_count = -(-time_steps // chunk_size)
    pad_size = chunk_count * chunk_size - time_steps
    padded = torch.nn.functional.pad(x, (0, pad_size))
    chunks = padded.reshape(batch_size, n_states, chunk_count, chunk_size)
    valid = torch.arange(chunk_count * chunk_size, device=x.device) < time_steps
    valid = valid.reshape(1, 1, chunk_count, chunk_size)
    counts = valid.sum(dim=-1).to(x.dtype)
    zeros = torch.zeros_like(chunks)
    means = torch.where(valid, chunks, zeros).sum(dim=-1) / counts
    squared_deviations = torch.where(valid, (chunks - means.unsqueeze(-1)).square(), zeros)
    variances = squared_deviations.sum(dim=-1) / (counts - 1.0)
    minima = torch.where(valid, chunks, torch.full_like(chunks, float("inf"))).amin(dim=-1)
    maxima = torch.where(valid, chunks, torch.full_like(chunks, float("-inf"))).amax(dim=-1)
    return torch.stack((means, variances, minima, maxima), dim=-1)


class AggLinearTrend(Feature[AggLinearTrendParams]):
    r"""Linear trend of per-chunk aggregate values.

    The series is split into consecutive chunks of ``chunk_size`` samples, retaining a shorter
    final chunk. Each chunk is reduced with ``f_agg``, then regressed against its zero-based
    chunk index. A chunk size at least as large as the series produces NaN.

    Args:
        params: One dict per requested chunk size, aggregate, and statistic. Defaults to the
            comprehensive tstorch configuration when omitted. An explicit empty list is
            rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[AggLinearTrendParams] | None = None) -> None:
        if params is None:
            params = list(_AGG_LINEAR_TREND_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "AggLinearTrend requires at least one param; omit `params` to use the "
                f"default {_AGG_LINEAR_TREND_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[AggLinearTrendParams] = params
        grouped: dict[int, list[tuple[int, AggLinearTrendParams]]] = {}
        for output_index, param in enumerate(params):
            grouped.setdefault(param["chunk_size"], []).append((output_index, param))

        groups: list[tuple[int, Tensor]] = []
        grouped_output_order: list[int] = []
        for chunk_size, entries in grouped.items():
            groups.append((
                chunk_size,
                torch.tensor(
                    [
                        [_AGGREGATE_INDEX[param["f_agg"]], _TREND_ATTR_INDEX[param["attr"]]]
                        for _, param in entries
                    ],
                    dtype=torch.long,
                ),
            ))
            grouped_output_order.extend(output_index for output_index, _ in entries)
        self._groups = tuple(groups)
        self._restore_order = torch.tensor(
            [grouped_output_order.index(index) for index in range(len(params))],
            dtype=torch.long,
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Each structural chunk partition is built once and shared by all matching params.
        group_results: list[Tensor] = []
        for chunk_size, indices in self._groups:
            output_count = indices.shape[0]
            if chunk_size >= x.shape[-1]:
                group_results.append(
                    torch.full(
                        (*x.shape[:2], output_count),
                        float("nan"),
                        dtype=x.dtype,
                        device=x.device,
                    )
                )
                continue
            aggregates = _aggregate_chunks(x, chunk_size)
            aggregate_indices = indices[:, 0].to(x.device)
            selected_aggregates = aggregates.gather(
                -1,
                aggregate_indices.view(1, 1, 1, output_count).expand(
                    *aggregates.shape[:-1], output_count
                ),
            )
            stats = linregress(selected_aggregates.movedim(-1, -2))
            attr_indices = indices[:, 1].to(x.device)
            selected_stats = stats.gather(
                -1,
                attr_indices.view(1, 1, output_count, 1).expand(*stats.shape[:-1], 1),
            ).squeeze(-1)
            group_results.append(selected_stats.to(x.dtype))
        grouped_results = torch.cat(group_results, dim=-1)
        return grouped_results.index_select(-1, self._restore_order.to(x.device))

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format:
            ``agg_linear_trend__chunk_size_<chunk_size>__f_agg_<f_agg>__attr_<attr>``.
        """
        return [
            "agg_linear_trend"
            f"__chunk_size_{param['chunk_size']}"
            f"__f_agg_{param['f_agg']}"
            f"__attr_{param['attr']}"
            for param in self.params
        ]


class ArCoefficient(Feature[ArCoefficientParams]):
    r"""Coefficient of an autoregressive model fitted with an intercept.

    The conditional maximum-likelihood model is
    \(x_t = \varphi_0 + \sum_{i=1}^{k}\varphi_i x_{t-i} + \varepsilon_t\).
    ``coeff=0`` selects the intercept and ``coeff=i`` selects lag coefficient \(i\).
    A coefficient above the fitted order is NaN.

    Args:
        params: One dict per requested model order and coefficient. Defaults to coefficients
            0 through 10 of an AR(10) model when omitted. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[ArCoefficientParams] | None = None) -> None:
        if params is None:
            params = list(_AR_COEFFICIENT_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "ArCoefficient requires at least one param; omit `params` to use the "
                f"default {_AR_COEFFICIENT_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[ArCoefficientParams] = params
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, param in enumerate(params):
            grouped.setdefault(param["k"], []).append((output_index, param["coeff"]))
        self._groups: tuple[tuple[int, Tensor], ...] = tuple(
            (
                order,
                torch.tensor([coefficient for _, coefficient in entries], dtype=torch.long),
            )
            for order, entries in grouped.items()
        )
        grouped_output_order = [
            output_index for entries in grouped.values() for output_index, _ in entries
        ]
        self._restore_order = torch.tensor(
            [grouped_output_order.index(index) for index in range(len(params))],
            dtype=torch.long,
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Each fitted order is shared by every requested coefficient at that order.
        group_results: list[Tensor] = []
        for order, coefficients in self._groups:
            parameters = auto_reg(x, order)
            valid = coefficients <= order
            selected = parameters.gather(
                -1,
                coefficients.clamp(max=order).to(x.device).view(1, 1, -1).expand(*x.shape[:2], -1),
            )
            group_results.append(
                torch.where(
                    valid.to(x.device).view(1, 1, -1),
                    selected,
                    torch.full_like(selected, float("nan")),
                )
            )
        grouped_results = torch.cat(group_results, dim=-1)
        return grouped_results.index_select(-1, self._restore_order.to(x.device))

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``ar_coefficient__k_<k>__coeff_<coeff>``.
        """
        return [f"ar_coefficient__k_{param['k']}__coeff_{param['coeff']}" for param in self.params]


class AugmentedDickeyFuller(Feature[AugmentedDickeyFullerParams]):
    r"""Augmented Dickey-Fuller unit-root test.

    The constant-only ADF regression selects a lag using AIC, BIC, the final-lag t-statistic,
    or the full default lag order. The p-value uses MacKinnon's regression-surface
    approximation. Constant or too-short series produce NaN.

    Args:
        params: One dict per requested statistic and lag-selection method. Omitted ``autolag``
            values use AIC. Defaults to all three AIC results when omitted. An explicit empty
            list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[AugmentedDickeyFullerParams] | None = None) -> None:
        if params is None:
            params = list(_ADF_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "AugmentedDickeyFuller requires at least one param; omit `params` to use the "
                f"default {_ADF_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[AugmentedDickeyFullerParams] = params
        grouped: dict[Autolag, list[tuple[int, AugmentedDickeyFullerParams]]] = {}
        for output_index, param in enumerate(params):
            if "autolag" in param:
                autolag: Autolag = param["autolag"]
            else:
                autolag = "AIC"
            grouped.setdefault(autolag, []).append((output_index, param))
        self._groups: tuple[tuple[Autolag, Tensor], ...] = tuple(
            (
                autolag,
                torch.tensor(
                    [_ADF_ATTR_INDEX[param["attr"]] for _, param in entries],
                    dtype=torch.long,
                ),
            )
            for autolag, entries in grouped.items()
        )
        grouped_output_order = [
            output_index for entries in grouped.values() for output_index, _ in entries
        ]
        self._restore_order = torch.tensor(
            [grouped_output_order.index(index) for index in range(len(params))],
            dtype=torch.long,
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Each autolag regression is shared by every requested result attribute.
        group_results = [
            _select_columns(adfuller(x, autolag), attr_indices)
            for autolag, attr_indices in self._groups
        ]
        grouped_results = torch.cat(group_results, dim=-1)
        return grouped_results.index_select(-1, self._restore_order.to(x.device))

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format:
            ``augmented_dickey_fuller__attr_<attr>__autolag_<autolag>``.
        """
        return [
            f"augmented_dickey_fuller__attr_{param['attr']}__autolag_{param.get('autolag', 'AIC')}"
            for param in self.params
        ]
