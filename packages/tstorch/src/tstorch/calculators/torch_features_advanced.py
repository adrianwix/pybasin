"""Advanced feature calculators for ``(B, S, T)`` time-series tensors."""

import math
from typing import TypedDict

import torch
from torch import Tensor

from tstorch.feature_factory import Feature, ParameterlessFeature


class BenfordCorrelation(ParameterlessFeature):
    r"""Correlation between leading-digit frequencies and Benford's law.

    For digits \(d = 1, \ldots, 9\), Benford's expected frequency is

    \[
    p_d = \log_{10}\left(1 + \frac{1}{d}\right).
    \]

    The observed frequencies use the full time-series length as their denominator.
    Zeros and NaNs therefore dilute every digit bin. Infinite values follow tsfresh's
    ``nan_to_num`` semantics and are replaced by the largest finite value of the input
    dtype before their leading digit is counted. A constant observed distribution has
    undefined correlation and returns NaN.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the Benford correlation.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1).
        """
        time_steps = x.shape[2]
        benford = x.new_tensor([math.log10(1 + 1 / digit) for digit in range(1, 10)])
        benford_centered = benford - benford.mean()
        benford_std = benford.std(correction=1)

        absolute_values = torch.nan_to_num(x).abs()
        positive = absolute_values > 0
        logarithms = torch.where(
            positive,
            absolute_values.log10(),
            torch.zeros_like(absolute_values),
        )
        powers = logarithms.floor()
        first_digits = torch.where(
            positive,
            (absolute_values / (10**powers)).floor().long().clamp(1, 9),
            torch.zeros_like(absolute_values, dtype=torch.long),
        )

        digit_bins = torch.arange(1, 10, device=x.device).view(1, 1, 1, 9)
        counts = (first_digits.unsqueeze(-1) == digit_bins).sum(dim=2)
        frequencies = counts.to(x.dtype) / time_steps
        frequencies_centered = frequencies - frequencies.mean(dim=-1, keepdim=True)
        frequencies_std = frequencies.std(dim=-1, correction=1)

        numerator = (frequencies_centered * benford_centered.view(1, 1, 9)).sum(dim=-1)
        denominator = frequencies_std * benford_std * 8
        return (numerator / denominator).unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``benford_correlation``.
        """
        return ["benford_correlation"]


class C3Params(TypedDict):
    """Parameters for one `C3` output column.

    Attributes:
        lag: Non-negative spacing between the three multiplied samples.
    """

    lag: int


_C3_DEFAULT_PARAMS: list[C3Params] = [{"lag": lag} for lag in range(1, 4)]


class C3(Feature[C3Params]):
    r"""Mean lagged triple product at every requested lag.

    \[
    c_3(k) = \frac{1}{T - 2k}
    \sum_{t=0}^{T - 2k - 1} x_t x_{t+k} x_{t+2k}.
    \]

    A lag with \(2k \ge T\) returns 0. Several requested lags are broadcast over one
    tensor operation; a single lag uses direct contiguous slices.

    Args:
        params: One dict per output, each containing ``lag``. Defaults to lags 1, 2,
            and 3. Passing an empty list raises ``ValueError``.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[C3Params] | None = None) -> None:
        if params is None:
            params = list(_C3_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "C3 requires at least one param; omit `params` to use the default "
                f"{_C3_DEFAULT_PARAMS!r} instead of passing []."
            )
        if any(param["lag"] < 0 for param in params):
            raise ValueError("C3 requires non-negative lag values.")
        self.params: list[C3Params] = params
        self._lags: list[int] = [param["lag"] for param in params]

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the requested C3 statistics.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Requested lags broadcast together; the single-lag path avoids the gather setup.
        if len(self._lags) == 1:
            first, second, third = _single_shifted_triples(x, self._lags[0])
            if first.shape[2] == 0:
                return torch.zeros(*x.shape[:2], 1, dtype=x.dtype, device=x.device)
            return (first * second * third).mean(dim=2, keepdim=True)

        first, second, third, valid = _shifted_triples(x, self._lags)
        products = torch.where(valid, first * second * third, torch.zeros_like(first))
        counts = valid.sum(dim=2).clamp(min=1)
        return products.sum(dim=2) / counts

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order, formatted as ``c3__lag_<lag>``.
        """
        return [f"c3__lag_{param['lag']}" for param in self.params]


class EnergyRatioByChunksParams(TypedDict):
    """Parameters for one `EnergyRatioByChunks` output column.

    Attributes:
        num_segments: Number of contiguous chunks in the partition.
        segment_focus: Zero-based chunk whose energy ratio is returned.
    """

    num_segments: int
    segment_focus: int


_ENERGY_RATIO_BY_CHUNKS_DEFAULT_PARAMS: list[EnergyRatioByChunksParams] = [
    {"num_segments": 10, "segment_focus": segment_focus} for segment_focus in range(10)
]


class EnergyRatioByChunks(Feature[EnergyRatioByChunksParams]):
    r"""Fraction of total energy contained in each requested chunk.

    The time axis is partitioned with ``numpy.array_split`` semantics. If the series
    length is not divisible by ``num_segments``, the first chunks receive one extra
    sample. For focused chunk \(i\), the result is

    \[
    \frac{\sum_{t \in \mathrm{chunk}_i} x_t^2}{\sum_{t=0}^{T-1} x_t^2}.
    \]

    An all-zero series returns NaN. Requests sharing ``num_segments`` reuse one squared
    cumulative sum; different structural partitions are evaluated once per distinct
    ``num_segments`` value.

    Args:
        params: One dict per output, containing ``num_segments`` and ``segment_focus``.
            Defaults to all ten chunks of a ten-segment partition. Passing an empty list
            raises ``ValueError``.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[EnergyRatioByChunksParams] | None = None) -> None:
        if params is None:
            params = list(_ENERGY_RATIO_BY_CHUNKS_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "EnergyRatioByChunks requires at least one param; omit `params` to use "
                f"the default {_ENERGY_RATIO_BY_CHUNKS_DEFAULT_PARAMS!r} instead of passing []."
            )
        for param in params:
            num_segments = param["num_segments"]
            segment_focus = param["segment_focus"]
            if num_segments <= 0:
                raise ValueError("EnergyRatioByChunks requires num_segments > 0.")
            if not 0 <= segment_focus < num_segments:
                raise ValueError("EnergyRatioByChunks requires 0 <= segment_focus < num_segments.")

        self.params: list[EnergyRatioByChunksParams] = params
        grouped_focuses: dict[int, list[int]] = {}
        column_positions: list[tuple[int, int]] = []
        for param in params:
            num_segments = param["num_segments"]
            focuses = grouped_focuses.setdefault(num_segments, [])
            column_positions.append((num_segments, len(focuses)))
            focuses.append(param["segment_focus"])
        self._grouped_focuses: dict[int, Tensor] = {
            num_segments: torch.tensor(focuses, dtype=torch.long)
            for num_segments, focuses in grouped_focuses.items()
        }
        self._column_positions: list[tuple[int, int]] = column_positions

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the requested chunk-energy ratios.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Each structural partition shares one cumulative-energy pass across its focuses.
        results = {
            num_segments: _energy_ratio_by_chunks(x, num_segments, focuses)
            for num_segments, focuses in self._grouped_focuses.items()
        }
        columns = [
            results[num_segments][..., position : position + 1]
            for num_segments, position in self._column_positions
        ]
        return torch.cat(columns, dim=-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format:
            ``energy_ratio_by_chunks__num_segments_<num_segments>__segment_focus_<segment_focus>``.
        """
        return [
            f"energy_ratio_by_chunks__num_segments_{param['num_segments']}"
            f"__segment_focus_{param['segment_focus']}"
            for param in self.params
        ]


class TimeReversalAsymmetryStatisticParams(TypedDict):
    """Parameters for one `TimeReversalAsymmetryStatistic` output column.

    Attributes:
        lag: Non-negative spacing between the three samples in each term.
    """

    lag: int


_TIME_REVERSAL_ASYMMETRY_STATISTIC_DEFAULT_PARAMS: list[TimeReversalAsymmetryStatisticParams] = [
    {"lag": lag} for lag in range(1, 4)
]


class TimeReversalAsymmetryStatistic(Feature[TimeReversalAsymmetryStatisticParams]):
    r"""Time-reversal asymmetry statistic at every requested lag.

    \[
    \frac{1}{T - 2k} \sum_{t=0}^{T - 2k - 1}
    \left(x_{t+2k}^2 x_{t+k} - x_{t+k} x_t^2\right).
    \]

    A lag with \(2k \ge T\) returns 0. Several requested lags are broadcast over one
    tensor operation; a single lag uses direct contiguous slices.

    Args:
        params: One dict per output, each containing ``lag``. Defaults to lags 1, 2,
            and 3. Passing an empty list raises ``ValueError``.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(
        self,
        params: list[TimeReversalAsymmetryStatisticParams] | None = None,
    ) -> None:
        if params is None:
            params = list(_TIME_REVERSAL_ASYMMETRY_STATISTIC_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "TimeReversalAsymmetryStatistic requires at least one param; omit `params` "
                "to use the default "
                f"{_TIME_REVERSAL_ASYMMETRY_STATISTIC_DEFAULT_PARAMS!r} instead of passing []."
            )
        if any(param["lag"] < 0 for param in params):
            raise ValueError("TimeReversalAsymmetryStatistic requires non-negative lag values.")
        self.params: list[TimeReversalAsymmetryStatisticParams] = params
        self._lags: list[int] = [param["lag"] for param in params]

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the requested time-reversal statistics.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Requested lags broadcast together; the single-lag path avoids the gather setup.
        if len(self._lags) == 1:
            first, second, third = _single_shifted_triples(x, self._lags[0])
            if first.shape[2] == 0:
                return torch.zeros(*x.shape[:2], 1, dtype=x.dtype, device=x.device)
            terms = third**2 * second - second * first**2
            return terms.mean(dim=2, keepdim=True)

        first, second, third, valid = _shifted_triples(x, self._lags)
        terms = third**2 * second - second * first**2
        terms = torch.where(valid, terms, torch.zeros_like(terms))
        counts = valid.sum(dim=2).clamp(min=1)
        return terms.sum(dim=2) / counts

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order, formatted as
            ``time_reversal_asymmetry_statistic__lag_<lag>``.
        """
        return [f"time_reversal_asymmetry_statistic__lag_{param['lag']}" for param in self.params]


def _single_shifted_triples(x: Tensor, lag: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return direct contiguous views for one lag."""
    valid_length = max(x.shape[2] - 2 * lag, 0)
    return (
        x[..., :valid_length],
        x[..., lag : lag + valid_length],
        x[..., 2 * lag : 2 * lag + valid_length],
    )


def _shifted_triples(x: Tensor, lags: list[int]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Broadcast shifted triples and their validity mask over several lags."""
    batch_size, n_states, time_steps = x.shape
    lag_tensor = torch.tensor(lags, device=x.device, dtype=torch.long)
    n_lags = len(lags)

    time_indices = torch.arange(time_steps, device=x.device).view(1, 1, time_steps, 1)
    lag_view = lag_tensor.view(1, 1, 1, n_lags)
    valid = time_indices + 2 * lag_view < time_steps

    once_shifted_indices = (time_indices + lag_view).clamp(max=time_steps - 1)
    twice_shifted_indices = (time_indices + 2 * lag_view).clamp(max=time_steps - 1)
    expanded_shape = (batch_size, n_states, time_steps, n_lags)
    expanded = x.unsqueeze(-1).expand(expanded_shape)
    once_shifted = expanded.gather(dim=2, index=once_shifted_indices.expand(expanded_shape))
    twice_shifted = expanded.gather(dim=2, index=twice_shifted_indices.expand(expanded_shape))
    return expanded, once_shifted, twice_shifted, valid.expand(expanded_shape)


def _energy_ratio_by_chunks(x: Tensor, num_segments: int, segment_focuses: Tensor) -> Tensor:
    """Return focused chunk-energy ratios for one fixed partition."""
    time_steps = x.shape[2]
    focuses = segment_focuses.to(device=x.device, dtype=torch.long)

    base = time_steps // num_segments
    remainder = time_steps % num_segments
    in_larger_chunks = focuses < remainder
    larger_starts = focuses * (base + 1)
    smaller_starts = remainder * (base + 1) + (focuses - remainder) * base
    starts = torch.where(in_larger_chunks, larger_starts, smaller_starts)
    widths = torch.where(
        in_larger_chunks,
        torch.full_like(focuses, base + 1),
        torch.full_like(focuses, base),
    )
    ends = starts + widths

    cumulative_energy = torch.nn.functional.pad((x**2).cumsum(dim=2), (1, 0))
    total_energy = cumulative_energy[..., -1:]
    segment_energy = cumulative_energy.index_select(2, ends) - cumulative_energy.index_select(
        2, starts
    )
    zero_energy = total_energy == 0
    ratios = segment_energy / torch.where(zero_energy, torch.ones_like(total_energy), total_energy)
    return torch.where(zero_energy, torch.full_like(ratios, float("nan")), ratios)
