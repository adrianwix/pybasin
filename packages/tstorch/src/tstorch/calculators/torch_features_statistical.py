"""Statistical feature calculators for time series.

All features follow a consistent tensor shape convention:
- Input: (B, S, T) where B=batch size, S=state variables, T=timesteps
- Output: (B, S, F) always, where F=1 for single-output features and F=len(params)
  for multi-output features (e.g. `Quantile`).

Features are computed along the time dimension (dim=2), preserving batch and state
dimensions. Each class below satisfies the ``Feature`` protocol
(``tstorch.feature_factory.Feature``): direct calls and ``TorchFeatureExtractor`` both
construct the same class the same way, there is no separate public/internal function pair.
"""

from typing import TypedDict

import torch
from torch import Tensor

from tstorch.feature_factory import Feature, ParameterlessFeature


def _median(x: Tensor) -> Tensor:
    """Shared median kernel behind `Median` and `SymmetryLooking`.

    Sorts the series and, for an even number of timesteps, averages the two middle
    elements - matching ``numpy.median`` (and tsfresh, which uses it). This differs
    from ``torch.median``, which returns the *lower* of the two middle values for an
    even-length input, and from ``torch.quantile(x, 0.5)``, whose CPU kernel is not
    traceable under ``torch.compile(dynamic=True)`` with symbolic shapes.
    """
    T = x.shape[2]
    sorted_x = x.sort(dim=2).values
    if T % 2 == 1:
        return sorted_x[:, :, T // 2 : T // 2 + 1]
    lo = sorted_x[:, :, T // 2 - 1 : T // 2]
    hi = sorted_x[:, :, T // 2 : T // 2 + 1]
    return (lo + hi) / 2


def _delta(x: Tensor) -> Tensor:
    """Shared kernel behind `Delta` and `LogDelta`: absolute difference between the
    maximum and the mean of the time series."""
    return (x.max(dim=2, keepdim=True).values - x.mean(dim=2, keepdim=True)).abs()


def _zero_out_fperr(t: Tensor) -> Tensor:
    """Treat |t| < 1e-14 as exactly zero, matching pandas' ``nanops._zero_out_fperr``.

    Applied to the raw central-moment sums in `Kurtosis`/`Skewness` since tsfresh's
    ``kurtosis``/``skewness`` wrap ``pandas.Series.kurt``/``.skew``, which apply this
    rule before dividing.
    """
    return torch.where(t.abs() < 1e-14, torch.zeros_like(t), t)


# =============================================================================
# MINIMAL FEATURES (tsfresh MinimalFCParameters - 10 features)
# =============================================================================


class SumValues(ParameterlessFeature):
    """Sum of all values along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.sum(dim=2, keepdim=True)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``sum_values``.
        """
        return ["sum_values"]


class Median(ParameterlessFeature):
    """Median of the time series along the time dimension.

    Sorts the series and, for an even number of timesteps, averages the two middle
    elements - matching ``numpy.median`` (and tsfresh, which uses it). This differs
    from ``torch.median``, which returns the *lower* of the two middle values for an
    even-length input.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return _median(x)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``median``.
        """
        return ["median"]


class Mean(ParameterlessFeature):
    """Mean of the time series along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.mean(dim=2, keepdim=True)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``mean``.
        """
        return ["mean"]


class Length(ParameterlessFeature):
    """Length of the time series (number of timesteps).

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        T = x.shape[2]
        return torch.full((*x.shape[:2], 1), T, dtype=x.dtype, device=x.device)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``length``.
        """
        return ["length"]


class StandardDeviation(ParameterlessFeature):
    """Standard deviation (population, ddof=0) along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.std(dim=2, correction=0, keepdim=True)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``standard_deviation``.
        """
        return ["standard_deviation"]


class Variance(ParameterlessFeature):
    """Variance (population, ddof=0) along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.var(dim=2, correction=0, keepdim=True)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``variance``.
        """
        return ["variance"]


class RootMeanSquare(ParameterlessFeature):
    """Root mean square value along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return torch.sqrt((x**2).mean(dim=2, keepdim=True))

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``root_mean_square``.
        """
        return ["root_mean_square"]


class Maximum(ParameterlessFeature):
    """Maximum value along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.max(dim=2, keepdim=True).values

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``maximum``.
        """
        return ["maximum"]


class AbsoluteMaximum(ParameterlessFeature):
    """Maximum absolute value along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.abs().max(dim=2, keepdim=True).values

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``absolute_maximum``.
        """
        return ["absolute_maximum"]


class Minimum(ParameterlessFeature):
    """Minimum value along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.min(dim=2, keepdim=True).values

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``minimum``.
        """
        return ["minimum"]


class Delta(ParameterlessFeature):
    """Absolute difference between the maximum and the mean of the time series.

    Has no tsfresh equivalent.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return _delta(x)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``delta``.
        """
        return ["delta"]


class LogDelta(ParameterlessFeature):
    """Natural logarithm of `Delta`.

    A small epsilon (``1e-10``) is added before taking the log to avoid ``-inf`` when
    the delta is exactly zero. Has no tsfresh equivalent.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return torch.log(_delta(x) + 1e-10)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``log_delta``.
        """
        return ["log_delta"]


# =============================================================================
# SIMPLE STATISTICS (5 features)
# =============================================================================


class AbsEnergy(ParameterlessFeature):
    r"""Absolute energy (sum of squares) along the time dimension.

    \[
    E = \sum_{i=1}^{T} x_i^2
    \]

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return (x**2).sum(dim=2, keepdim=True)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``abs_energy``.
        """
        return ["abs_energy"]


class Kurtosis(ParameterlessFeature):
    r"""Fisher's (excess) kurtosis along the time dimension.

    \[
    G_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \frac{\sum_i (x_i - \bar{x})^4}
    {\left(\frac{\sum_i (x_i - \bar{x})^2}{n-1}\right)^2}
    - \frac{3(n-1)^2}{(n-2)(n-3)}
    \]

    Computed via ``pandas.core.nanops.nankurt`` (what tsfresh's ``kurtosis`` wraps),
    which accumulates raw sums of powers in float64 and zeroes out sums with magnitude
    below ``1e-14`` before dividing, rather than the textbook mean-of-central-moments
    formula: for a near-constant series (state std many orders of magnitude below its
    mean), a mean-based zeroing threshold and pandas' sum-based one land in different
    places, so matching pandas' exact algorithm - not just switching to float64
    accumulation - is what matches tsfresh's reference values.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1). NaN when ``T < 4``.
        """
        count = x.shape[2]
        x64 = x.double()
        mean = x64.sum(dim=2, keepdim=True) / count
        adjusted = x64 - mean
        m2 = _zero_out_fperr((adjusted**2).sum(dim=2, keepdim=True))
        m4 = (adjusted**4).sum(dim=2, keepdim=True)

        adj = 3 * (count - 1) ** 2 / ((count - 2) * (count - 3))
        numerator = _zero_out_fperr(count * (count + 1) * (count - 1) * m4)
        denominator = _zero_out_fperr((count - 2) * (count - 3) * m2**2)

        is_degenerate = denominator == 0
        denominator_safe = torch.where(is_degenerate, torch.ones_like(denominator), denominator)
        result = numerator / denominator_safe - adj
        result = torch.where(is_degenerate, torch.zeros_like(result), result)
        if count < 4:
            result = torch.full_like(result, float("nan"))

        return result.to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``kurtosis``.
        """
        return ["kurtosis"]


class Skewness(ParameterlessFeature):
    r"""Fisher-Pearson adjusted skewness along the time dimension.

    \[
    G_1 = \frac{\sqrt{n(n-1)}}{n-2} \frac{\sum_i (x_i - \bar{x})^3 / n}
    {\left(\sum_i (x_i - \bar{x})^2 / n\right)^{3/2}}
    \]

    Computed via ``pandas.core.nanops.nanskew`` (what tsfresh's ``skewness`` wraps),
    which accumulates raw sums of powers in float64 and zeroes out sums with magnitude
    below ``1e-14`` before dividing, rather than the textbook mean-of-central-moments
    formula, for the same near-constant-series reason as `Kurtosis` above.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1). NaN when ``T < 3``.
        """
        count = x.shape[2]
        x64 = x.double()
        mean = x64.sum(dim=2, keepdim=True) / count
        adjusted = x64 - mean
        m2 = _zero_out_fperr((adjusted**2).sum(dim=2, keepdim=True))
        m3 = _zero_out_fperr((adjusted**3).sum(dim=2, keepdim=True))

        is_degenerate = m2 == 0
        m2_safe = torch.where(is_degenerate, torch.ones_like(m2), m2)
        result = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2_safe**1.5)
        result = torch.where(is_degenerate, torch.zeros_like(result), result)
        if count < 3:
            result = torch.full_like(result, float("nan"))
        return result.to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``skewness``.
        """
        return ["skewness"]


class QuantileParams(TypedDict):
    """Parameters for one `Quantile` output column.

    Attributes:
        q: Quantile to compute, in ``[0, 1]``.
    """

    q: float


# Matches `TORCH_COMPREHENSIVE_FC_PARAMETERS["quantile"]` in settings.py.
_QUANTILE_DEFAULT_PARAMS: list[QuantileParams] = [
    {"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
]


class Quantile(Feature[QuantileParams]):
    """Requested quantile(s) of the time series along the time dimension.

    Matches tsfresh's ``quantile`` (``np.quantile(x, q)``, linear interpolation between
    the two nearest ranks). Uses sort + gather-based linear interpolation rather than
    ``torch.quantile(x, q_tensor, dim=2)`` directly: with a vector ``q``, that call's CPU
    kernel is not traceable under ``torch.compile(fullgraph=True)`` (data-dependent
    operator break), the same class of issue `Median` avoids for the single q=0.5 case.

    tsfresh calls this once *per* q (``fctype="simple"``), independently re-sorting the
    series each time. This class sorts once and reads every requested q off the same
    sorted series via linear interpolation, so batching q's here is a genuine
    algorithmic improvement over tsfresh's own approach, not just a faster port of the
    same computation.

    Args:
        params: One dict per requested output, each with a ``q`` (float, in ``[0, 1]``)
            key. May repeat a value. Defaults to tsfresh's own default (8 quantiles from
            0.1 to 0.9) when omitted; an explicit empty list is rejected rather than
            silently producing a feature with no output columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[QuantileParams] | None = None) -> None:
        if params is None:
            params = list(_QUANTILE_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "Quantile requires at least one param; omit `params` to use the default "
                f"{_QUANTILE_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[QuantileParams] = params
        self._qs: Tensor = torch.tensor([p["q"] for p in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the requested quantiles.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        batch_size, n_states, T = x.shape
        K = self._qs.shape[0]  # always >= 1, guaranteed by __init__

        sorted_x = x.sort(dim=2).values

        q_tensor = self._qs.to(dtype=x.dtype, device=x.device)
        index = q_tensor * (T - 1)
        lower = index.floor().long()
        upper = index.ceil().long()
        frac = (index - lower.to(x.dtype)).view(1, 1, K)

        lower_idx = lower.clamp(0, T - 1).view(1, 1, K).expand(batch_size, n_states, K)
        upper_idx = upper.clamp(0, T - 1).view(1, 1, K).expand(batch_size, n_states, K)
        lower_vals = sorted_x.gather(dim=2, index=lower_idx)
        upper_vals = sorted_x.gather(dim=2, index=upper_idx)
        return lower_vals + frac * (upper_vals - lower_vals)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format: ``quantile__q_<q>``.
        """
        return [f"quantile__q_{p['q']}" for p in self.params]


class VariationCoefficient(ParameterlessFeature):
    """Coefficient of variation (``std(x) / mean(x)``) along the time dimension.

    Signed: unlike some definitions of the coefficient of variation, the mean is not
    taken in absolute value, so the result is negative whenever the series mean is
    negative - matching tsfresh exactly.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1). NaN when the mean is exactly 0.
        """
        mu = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, correction=0, keepdim=True)
        return torch.where(mu == 0, torch.full_like(mu, float("nan")), std / mu)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``variation_coefficient``.
        """
        return ["variation_coefficient"]


class MeanNAbsoluteMaxParams(TypedDict):
    """Parameters for one `MeanNAbsoluteMax` output column.

    Attributes:
        number_of_maxima: Number of largest absolute values to average.
    """

    number_of_maxima: int


# Matches `TORCH_COMPREHENSIVE_FC_PARAMETERS["mean_n_absolute_max"]` in settings.py.
_MEAN_N_ABSOLUTE_MAX_DEFAULT_PARAMS: list[MeanNAbsoluteMaxParams] = [
    {"number_of_maxima": n} for n in [3, 5, 7]
]


class MeanNAbsoluteMax(Feature[MeanNAbsoluteMaxParams]):
    """Mean of the ``number_of_maxima`` largest absolute values in the time series, for
    every requested ``number_of_maxima``.

    tsfresh calls this once *per* ``number_of_maxima`` (``fctype="simple"``),
    independently re-sorting the absolute values each time. This class sorts once (via
    ``topk``) and reads a cumulative mean at every requested ``number_of_maxima`` off
    that one sort, so batching here is a genuine algorithmic improvement over tsfresh's
    own approach.

    Args:
        params: One dict per requested output, each with a ``number_of_maxima`` (int)
            key. Defaults to tsfresh's own default (3, 5, and 7) when omitted; an
            explicit empty list is rejected rather than silently producing a feature with
            no output columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[MeanNAbsoluteMaxParams] | None = None) -> None:
        if params is None:
            params = list(_MEAN_N_ABSOLUTE_MAX_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "MeanNAbsoluteMax requires at least one param; omit `params` to use the "
                f"default {_MEAN_N_ABSOLUTE_MAX_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[MeanNAbsoluteMaxParams] = params
        self._ns: Tensor = torch.tensor([p["number_of_maxima"] for p in params], dtype=torch.long)
        self._max_n: int = max(p["number_of_maxima"] for p in params)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the requested means of largest-absolute-value counts.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)). NaN at any column whose
            ``number_of_maxima >= T`` (the series does not have that many distinct
            positions), a per-column condition since different columns can straddle
            the boundary independently for the same series.
        """
        batch_size, n_states, T = x.shape
        ns = self._ns.to(device=x.device, dtype=torch.long)
        K = ns.shape[0]  # always >= 1, guaranteed by __init__

        max_k = min(self._max_n, T)
        x_abs = x.abs()
        top_vals, _ = x_abs.topk(max_k, dim=2)  # (B, S, max_k), already sorted descending

        # Cumulative mean of the top-k values at every k in [1, max_k] in one pass, then gather
        # the requested k's directly instead of re-slicing/re-reducing per n in a Python loop.
        counts = torch.arange(1, max_k + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        running_mean = top_vals.cumsum(dim=2) / counts  # (B, S, max_k)

        ks = ns.clamp(max=max_k)
        gather_idx = (ks - 1).clamp(min=0).view(1, 1, -1).expand(batch_size, n_states, K)
        result = running_mean.gather(dim=2, index=gather_idx)

        is_nan_col = (ns >= T).view(1, 1, K)
        return torch.where(is_nan_col, torch.full_like(result, float("nan")), result)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format:
            ``mean_n_absolute_max__number_of_maxima_<number_of_maxima>``.
        """
        return [
            f"mean_n_absolute_max__number_of_maxima_{p['number_of_maxima']}" for p in self.params
        ]


class RatioBeyondRSigmaParams(TypedDict):
    """Parameters for one `RatioBeyondRSigma` output column.

    Attributes:
        r: Standard deviation multiplier.
    """

    r: float


# Matches `TORCH_COMPREHENSIVE_FC_PARAMETERS["ratio_beyond_r_sigma"]` in settings.py.
_RATIO_BEYOND_R_SIGMA_DEFAULT_PARAMS: list[RatioBeyondRSigmaParams] = [
    {"r": r} for r in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]
]


class RatioBeyondRSigma(Feature[RatioBeyondRSigmaParams]):
    r"""Fraction of values more than ``r`` standard deviations from the mean, for every
    requested ``r``.

    \[
    \frac{1}{T} \left| \{ i : |x_i - \bar{x}| > r \cdot \sigma \} \right|
    \]

    Computed in float64 internally, since this is a strict ``>`` comparison against a
    threshold derived from a reduction (``mean``/``std``), and values sitting almost
    exactly at the boundary are sensitive to summation-order rounding. tsfresh computes
    in the input's native (float32) precision; float64 can therefore disagree with
    tsfresh's reference value by up to one count out of T for values that fall within
    float32 rounding error of the ``r * sigma`` boundary.

    tsfresh calls this once *per* r (``fctype="simple"``), independently recomputing
    mean/std each time. This class computes mean/std once and broadcasts every
    requested r's threshold comparison against it, so batching r's here is a genuine
    algorithmic improvement over tsfresh's own approach.

    Args:
        params: One dict per requested output, each with an ``r`` (float) key. Defaults
            to tsfresh's own default (10 thresholds from 0.5 to 10) when omitted; an
            explicit empty list is rejected rather than silently producing a feature with
            no output columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[RatioBeyondRSigmaParams] | None = None) -> None:
        if params is None:
            params = list(_RATIO_BEYOND_R_SIGMA_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "RatioBeyondRSigma requires at least one param; omit `params` to use the "
                f"default {_RATIO_BEYOND_R_SIGMA_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[RatioBeyondRSigmaParams] = params
        self._rs: Tensor = torch.tensor([p["r"] for p in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the requested beyond-r-sigma ratios.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        rs = self._rs.to(device=x.device, dtype=torch.float64)  # rs.shape[0] always >= 1

        x64 = x.double()
        mu = x64.mean(dim=2, keepdim=True)
        std = x64.std(dim=2, correction=0, keepdim=True)
        deviation = (x64 - mu).abs().unsqueeze(-1)  # (B, S, T, 1)
        rs_view = rs.view(1, 1, 1, -1)
        beyond = deviation > rs_view * std.unsqueeze(-1)  # (B, S, T, K)
        return beyond.to(x.dtype).mean(dim=2)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format: ``ratio_beyond_r_sigma__r_<r>``.
        """
        return [f"ratio_beyond_r_sigma__r_{p['r']}" for p in self.params]


class SymmetryLookingParams(TypedDict):
    """Parameters for one `SymmetryLooking` output column.

    Attributes:
        r: Symmetry tolerance multiplier.
    """

    r: float


# Matches `TORCH_COMPREHENSIVE_FC_PARAMETERS["symmetry_looking"]` in settings.py.
_SYMMETRY_LOOKING_DEFAULT_PARAMS: list[SymmetryLookingParams] = [{"r": r * 0.05} for r in range(20)]


class SymmetryLooking(Feature[SymmetryLookingParams]):
    """Whether the time series distribution looks symmetric, for every requested ``r``.

    True (1.0) when ``|mean - median| < r * (max - min)``, i.e. the mean and median are
    close relative to the overall range of the series.

    Unlike `Quantile`/`MeanNAbsoluteMax`/`RatioBeyondRSigma` above, tsfresh itself calls
    ``symmetry_looking`` as ``fctype="combiner"`` (one call with a list of ``r`` dicts)
    and already shares the mean/median/range computation across every requested ``r``
    internally - so batching here is parity with tsfresh's own approach, not an
    additional algorithmic win.

    Args:
        params: One dict per requested output, each with an ``r`` (float) key. Defaults
            to tsfresh's own default (20 thresholds from 0.0 to 0.95) when omitted; an
            explicit empty list is rejected rather than silently producing a feature with
            no output columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[SymmetryLookingParams] | None = None) -> None:
        if params is None:
            params = list(_SYMMETRY_LOOKING_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "SymmetryLooking requires at least one param; omit `params` to use the "
                f"default {_SYMMETRY_LOOKING_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[SymmetryLookingParams] = params
        self._rs: Tensor = torch.tensor([p["r"] for p in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the requested symmetry-looking flags.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)), 1.0 or 0.0.
        """
        rs = self._rs.to(device=x.device, dtype=x.dtype)  # rs.shape[0] always >= 1

        mu = x.mean(dim=2, keepdim=True)
        med = _median(x)
        range_val = x.max(dim=2, keepdim=True).values - x.min(dim=2, keepdim=True).values
        rs_view = rs.view(1, 1, -1)
        return ((mu - med).abs() < rs_view * range_val).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format: ``symmetry_looking__r_<r>``.
        """
        return [f"symmetry_looking__r_{p['r']}" for p in self.params]


class Amplitude(ParameterlessFeature):
    """Peak-to-peak amplitude (max - min) of the time series.

    Has no tsfresh equivalent.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
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
        return x.max(dim=2, keepdim=True).values - x.min(dim=2, keepdim=True).values

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``amplitude``.
        """
        return ["amplitude"]
