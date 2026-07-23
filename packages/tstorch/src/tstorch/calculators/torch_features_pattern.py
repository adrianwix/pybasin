"""Streak and peak-pattern feature calculators."""

from __future__ import annotations

import math
from typing import TypedDict

import torch
from torch import Tensor
from torch.nn import functional

from tstorch.feature_factory import Feature, ParameterlessFeature


def _longest_consecutive_run(mask: Tensor) -> Tensor:
    """Return the longest true run along the final dimension."""
    batch_size, n_states, length = mask.shape
    if length == 0:
        return torch.zeros(batch_size, n_states, dtype=torch.long, device=mask.device)

    indices = torch.arange(length, device=mask.device).view(1, 1, length)
    reset_indices = torch.where(mask, indices.new_full((), -1), indices)
    most_recent_reset = torch.cummax(reset_indices, dim=2).values
    run_lengths = torch.where(mask, indices - most_recent_reset, indices.new_zeros(()))
    return run_lengths.amax(dim=2)


class LongestStrikeAboveMean(ParameterlessFeature):
    """Length of the longest consecutive run strictly above the series mean.

    Samples equal to the mean break a run. Empty and constant series return zero,
    matching tsfresh.

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
        mask = x > x.mean(dim=2, keepdim=True)
        return _longest_consecutive_run(mask).to(x.dtype).unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``longest_strike_above_mean``.
        """
        return ["longest_strike_above_mean"]


class LongestStrikeBelowMean(ParameterlessFeature):
    """Length of the longest consecutive run strictly below the series mean.

    Samples equal to the mean break a run. Empty and constant series return zero,
    matching tsfresh.

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
        mask = x < x.mean(dim=2, keepdim=True)
        return _longest_consecutive_run(mask).to(x.dtype).unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``longest_strike_below_mean``.
        """
        return ["longest_strike_below_mean"]


class NumberCrossingMParams(TypedDict):
    """Parameters for one `NumberCrossingM` output column.

    Attributes:
        m: Level whose crossings are counted.
    """

    m: float


_NUMBER_CROSSING_M_DEFAULT_PARAMS: list[NumberCrossingMParams] = [
    {"m": 0.0},
    {"m": -1.0},
    {"m": 1.0},
]


class NumberCrossingM(Feature[NumberCrossingMParams]):
    r"""Number of crossings of each requested level.

    For a level \(m\), a crossing occurs when two adjacent samples have different
    truth values under \(x_t > m\). A sample equal to \(m\) therefore belongs to the
    lower side of the crossing. Multiple levels are evaluated together by
    broadcasting a parameter axis.

    Args:
        params: One dict per requested output, each with an ``m`` key. Defaults to
            tsfresh's comprehensive levels ``0``, ``-1``, and ``1`` when omitted.
            An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[NumberCrossingMParams] | None = None) -> None:
        if params is None:
            params = list(_NUMBER_CROSSING_M_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "NumberCrossingM requires at least one param; omit `params` to use the "
                f"default {_NUMBER_CROSSING_M_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[NumberCrossingMParams] = params
        self._levels = torch.tensor([param["m"] for param in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Independent levels broadcast through one comparison without a Python param loop.
        levels = self._levels.to(dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
        above = x.unsqueeze(-1) > levels
        return (above[:, :, 1:] != above[:, :, :-1]).sum(dim=2).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format: ``number_crossing_m__m_<m>``.
        """
        return [f"number_crossing_m__m_{param['m']}" for param in self.params]


class NumberCwtPeaksParams(TypedDict):
    """Parameters for one `NumberCwtPeaks` output column.

    Attributes:
        n: Number of integer Ricker-wavelet widths, beginning at one.
    """

    n: int


_NUMBER_CWT_PEAKS_DEFAULT_PARAMS: list[NumberCwtPeaksParams] = [{"n": 1}, {"n": 5}]


def _ricker_wavelet(points: int, width: int) -> Tensor:
    positions = torch.arange(points, dtype=torch.float64) - (points - 1.0) / 2.0
    width_tensor = torch.tensor(float(width), dtype=torch.float64)
    squared = positions.square()
    coefficient = 2.0 / (torch.sqrt(3.0 * width_tensor) * math.pi**0.25)
    return (
        coefficient
        * (1.0 - squared / width_tensor.square())
        * torch.exp(-squared / (2.0 * width_tensor.square()))
    )


def _relative_maxima(x: Tensor) -> Tensor:
    length = x.shape[-1]
    if length < 3:
        return torch.zeros_like(x)
    greater_than_left = (x[..., 1:-1] > x[..., :-2]).to(x.dtype)
    greater_than_right = (x[..., 1:-1] > x[..., 2:]).to(x.dtype)
    return functional.pad(greater_than_left * greater_than_right, (1, 1))


def _noise_percentile(
    row: Tensor,
    window_indices: Tensor,
    window_valid: Tensor,
    lower_rank: Tensor,
    upper_rank: Tensor,
    upper_weight: Tensor,
) -> Tensor:
    windows = row[:, window_indices]
    windows = torch.where(window_valid.unsqueeze(0), windows, torch.full_like(windows, torch.inf))
    ordered = windows.sort(dim=2).values
    n_rows = row.shape[0]
    lower_values = torch.gather(
        ordered, 2, lower_rank.view(1, -1, 1).expand(n_rows, -1, 1)
    ).squeeze(2)
    upper_values = torch.gather(
        ordered, 2, upper_rank.view(1, -1, 1).expand(n_rows, -1, 1)
    ).squeeze(2)
    return lower_values + (upper_values - lower_values) * upper_weight.unsqueeze(0)


def _count_cwt_ridges(
    cwt: Tensor,
    max_distances: Tensor,
    window_indices: Tensor,
    window_valid: Tensor,
    lower_rank: Tensor,
    upper_rank: Tensor,
    upper_weight: Tensor,
) -> Tensor:
    n_rows, width_count, length = cwt.shape
    # Strict interior maxima cannot occupy adjacent samples.
    slot_count = width_count * ((length + 1) // 2)
    slot_ids = torch.arange(slot_count, device=cwt.device).view(1, slot_count)
    positions = torch.arange(length, device=cwt.device).view(1, length)
    sentinel = slot_count

    maxima = _relative_maxima(cwt)
    created = torch.zeros((n_rows, slot_count), dtype=torch.bool, device=cwt.device)
    alive = torch.zeros((n_rows, slot_count), dtype=torch.bool, device=cwt.device)
    endpoints = torch.zeros((n_rows, slot_count), dtype=torch.long, device=cwt.device)
    gaps = torch.zeros_like(endpoints)
    ridge_lengths = torch.zeros_like(endpoints)
    first_rows = torch.zeros_like(endpoints)
    first_cols = torch.zeros_like(endpoints)
    next_slot = torch.zeros((n_rows, 1), dtype=torch.long, device=cwt.device)

    for row_index in range(width_count - 1, -1, -1):
        active_sources = torch.where(alive, slot_ids, torch.full_like(slot_ids, sentinel))
        active_endpoints = torch.where(
            alive, endpoints.clamp(min=0, max=length - 1), torch.zeros_like(endpoints)
        )
        occupants = torch.full(
            (n_rows, length), sentinel, dtype=torch.long, device=cwt.device
        ).scatter_reduce(1, active_endpoints, active_sources, reduce="amin", include_self=True)

        occupied = occupants < sentinel
        left_positions = torch.cummax(
            torch.where(occupied, positions, torch.full_like(positions, -1)), dim=1
        ).values
        right_positions = torch.flip(
            torch.cummin(
                torch.flip(
                    torch.where(occupied, positions, torch.full_like(positions, length)),
                    dims=(1,),
                ),
                dim=1,
            ).values,
            dims=(1,),
        )

        left_owner = torch.gather(occupants, 1, left_positions.clamp(min=0))
        right_owner = torch.gather(occupants, 1, right_positions.clamp(max=length - 1))
        left_distance = torch.where(
            left_positions >= 0,
            positions - left_positions,
            torch.full_like(positions, length + 1),
        )
        right_distance = torch.where(
            right_positions < length,
            right_positions - positions,
            torch.full_like(positions, length + 1),
        )
        choose_left = (left_distance < right_distance) | (
            (left_distance == right_distance) & (left_owner <= right_owner)
        )
        owners = torch.where(choose_left, left_owner, right_owner)
        distances = torch.where(choose_left, left_distance, right_distance)

        row_maxima = maxima[:, row_index] > 0.5
        attached = (
            row_maxima
            & (owners < sentinel)
            & (distances.to(max_distances.dtype) <= max_distances[row_index])
        )
        safe_owners = owners.clamp(max=slot_count - 1)
        attachment_counts = torch.zeros_like(ridge_lengths).scatter_add(
            1, safe_owners, attached.to(torch.long)
        )
        attached_endpoints = torch.full_like(endpoints, -1).scatter_reduce(
            1,
            safe_owners,
            torch.where(attached, positions, torch.full_like(positions, -1)),
            reduce="amax",
            include_self=True,
        )
        has_attachment = attachment_counts > 0
        endpoints = torch.where(has_attachment, attached_endpoints, endpoints)
        ridge_lengths = ridge_lengths + attachment_counts
        first_rows = torch.where(has_attachment, row_index, first_rows)
        first_cols = torch.where(has_attachment, attached_endpoints, first_cols)
        gaps = torch.where(has_attachment, torch.zeros_like(gaps), gaps + alive.to(torch.long))
        alive = alive & (gaps <= 1)

        new_maxima = row_maxima & ~attached
        new_ranks = torch.cumsum(new_maxima.to(torch.long), dim=1) - 1
        new_slots = next_slot + new_ranks
        safe_new_slots = new_slots.clamp(min=0, max=slot_count - 1)
        new_columns = torch.full_like(endpoints, -1).scatter_reduce(
            1,
            safe_new_slots,
            torch.where(new_maxima, positions, torch.full_like(positions, -1)),
            reduce="amax",
            include_self=True,
        )
        new_created = new_columns >= 0
        created = created | new_created
        alive = alive | new_created
        endpoints = torch.where(new_created, new_columns, endpoints)
        ridge_lengths = torch.where(new_created, torch.ones_like(ridge_lengths), ridge_lengths)
        first_rows = torch.where(new_created, row_index, first_rows)
        first_cols = torch.where(new_created, new_columns, first_cols)
        gaps = torch.where(new_created, torch.zeros_like(gaps), gaps)
        next_slot = next_slot + new_maxima.sum(dim=1, keepdim=True)

    noises = _noise_percentile(
        cwt[:, 0],
        window_indices,
        window_valid,
        lower_rank,
        upper_rank,
        upper_weight,
    )
    row_offsets = torch.arange(n_rows, device=cwt.device).view(n_rows, 1) * width_count * length
    coefficient_indices = row_offsets + first_rows * length + first_cols
    signal = torch.gather(cwt.reshape(-1), 0, coefficient_indices.reshape(-1)).view_as(first_rows)
    noise = torch.gather(noises, 1, first_cols)
    snr = torch.abs(signal / noise)
    minimum_length = math.ceil(width_count / 4)
    accepted = created & (ridge_lengths >= minimum_length) & ~(snr < 1.0)
    return accepted.sum(dim=1)


class NumberCwtPeaks(Feature[NumberCwtPeaksParams]):
    r"""Count peaks found through Ricker-wavelet ridge lines.

    For each requested \(n\), the series is convolved with normalized Ricker
    wavelets at integer widths \(1,\ldots,n\). Strict local maxima in the resulting
    coefficient rows are joined into ridge lines when their displacement is no more
    than one quarter of the current width. A ridge may skip one width. SciPy's
    default filters are then applied: length at least \(\lceil n/4\rceil\), and
    signal-to-noise ratio at least one using the local tenth-percentile noise floor.

    A width-\(w\) Ricker wavelet is sampled as
    \(2(\sqrt{3w}\pi^{1/4})^{-1}(1-t^2/w^2)\exp(-t^2/(2w^2))\), using SciPy's
    truncation at \(\min(10w, T)\) samples. All widths are placed in one filter bank,
    and all requested outputs share one ``conv1d`` call at the largest width. Ridge
    topology depends on the final width, so each distinct ``n`` processes its own
    prefix of that transform.

    Floating-point convolution is backend-dependent. Coefficients separated only by
    roundoff can change a strict maximum and therefore an integer peak count. For
    example, a length-31 all-ones series returns zero for ``n=5`` here while SciPy
    1.17 returns one.

    Args:
        params: One dict per requested output, each with a positive ``n`` key.
            Defaults to tsfresh's comprehensive values ``1`` and ``5`` when omitted.
            An explicit empty list or non-positive value is rejected.

    """

    def __init__(self, params: list[NumberCwtPeaksParams] | None = None) -> None:
        if params is None:
            params = list(_NUMBER_CWT_PEAKS_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "NumberCwtPeaks requires at least one param; omit `params` to use the "
                f"default {_NUMBER_CWT_PEAKS_DEFAULT_PARAMS!r} instead of passing []."
            )
        if any(param["n"] <= 0 for param in params):
            raise ValueError("NumberCwtPeaks requires every `n` to be positive.")
        self.params: list[NumberCwtPeaksParams] = params
        self._width_counts = tuple(param["n"] for param in params)
        self._max_width_count = max(self._width_counts)
        kernel_length = 10 * self._max_width_count
        wavelet_bank = torch.zeros((self._max_width_count, kernel_length), dtype=torch.float64)
        global_left_padding = kernel_length // 2
        for index, width in enumerate(range(1, self._max_width_count + 1)):
            kernel = _ricker_wavelet(10 * width, width)
            offset = global_left_padding - kernel.shape[0] // 2
            wavelet_bank[index, offset : offset + kernel.shape[0]] = kernel
        self._wavelet_bank = wavelet_bank
        self._max_distances = torch.arange(1, self._max_width_count + 1, dtype=torch.float64) / 4.0

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T), with at least one time sample.

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        batch_size, n_states, length = x.shape
        if length == 0:
            raise ValueError("NumberCwtPeaks requires at least one time sample.")

        kernel_length = min(10 * self._max_width_count, length)
        if kernel_length == self._wavelet_bank.shape[1]:
            wavelet_bank = self._wavelet_bank.to(dtype=x.dtype, device=x.device)
        else:
            kernels: list[Tensor] = []
            global_left_padding = kernel_length // 2
            for width in range(1, self._max_width_count + 1):
                points = min(10 * width, length)
                kernel = _ricker_wavelet(points, width).to(dtype=x.dtype, device=x.device)
                offset = global_left_padding - points // 2
                kernels.append(functional.pad(kernel, (offset, kernel_length - points - offset)))
            wavelet_bank = torch.stack(kernels)

        rows = x.reshape(batch_size * n_states, 1, length)
        padded = functional.pad(rows, (kernel_length // 2, (kernel_length - 1) // 2))
        cwt = functional.conv1d(padded, wavelet_bank.unsqueeze(1))

        window_size = math.ceil(length / 20)
        half_window, odd = divmod(window_size, 2)
        positions = torch.arange(length, device=x.device)
        starts = (positions - half_window).clamp(min=0)
        ends = (positions + half_window + odd).clamp(max=length)
        sizes = ends - starts
        offsets = torch.arange(window_size, device=x.device).view(1, -1)
        window_indices = (starts.view(-1, 1) + offsets).clamp(max=length - 1)
        window_valid = offsets < sizes.view(-1, 1)
        percentile_positions = (sizes.to(torch.float64) - 1.0) * 0.1
        lower_rank = percentile_positions.floor().to(torch.long)
        upper_rank = percentile_positions.ceil().to(torch.long)
        upper_weight = percentile_positions - lower_rank

        # Widths share the CWT, while each requested prefix defines a different ridge topology.
        counts = [
            _count_cwt_ridges(
                cwt[:, :width_count],
                self._max_distances[:width_count].to(device=x.device),
                window_indices,
                window_valid,
                lower_rank,
                upper_rank,
                upper_weight,
            )
            for width_count in self._width_counts
        ]
        return torch.stack(counts, dim=1).to(x.dtype).view(batch_size, n_states, -1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format: ``number_cwt_peaks__n_<n>``.
        """
        return [f"number_cwt_peaks__n_{param['n']}" for param in self.params]


class NumberPeaksParams(TypedDict):
    """Parameters for one `NumberPeaks` output column.

    Attributes:
        n: Number of neighbours required on each side of a peak.
    """

    n: int


_NUMBER_PEAKS_DEFAULT_PARAMS: list[NumberPeaksParams] = [
    {"n": 1},
    {"n": 3},
    {"n": 5},
    {"n": 10},
    {"n": 50},
]


class NumberPeaks(Feature[NumberPeaksParams]):
    r"""Count strict peaks at every requested support.

    A sample at index \(t\) is a peak of support \(n\) when it is strictly greater
    than each of its \(n\) neighbours on both sides. The first and last \(n\)
    samples cannot be peaks.

    One window at the largest requested support supplies all outputs. Cumulative
    comparisons from the centre outwards then expose every smaller support without
    repeating the unfold.

    Args:
        params: One dict per requested output, each with a positive ``n`` key.
            Defaults to tsfresh's comprehensive supports ``1``, ``3``, ``5``, ``10``,
            and ``50`` when omitted. An explicit empty list or non-positive support is
            rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[NumberPeaksParams] | None = None) -> None:
        if params is None:
            params = list(_NUMBER_PEAKS_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "NumberPeaks requires at least one param; omit `params` to use the "
                f"default {_NUMBER_PEAKS_DEFAULT_PARAMS!r} instead of passing []."
            )
        if any(param["n"] <= 0 for param in params):
            raise ValueError("NumberPeaks requires every `n` to be positive.")
        self.params: list[NumberPeaksParams] = params
        self._supports = torch.tensor([param["n"] for param in params], dtype=torch.long)
        self._max_support = max(param["n"] for param in params)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # One maximal unfold shares neighbour comparisons across every support.
        batch_size, n_states, length = x.shape
        max_support = self._max_support
        supports = self._supports.to(device=x.device)
        padded = functional.pad(x, (max_support, max_support), value=float("-inf"))
        windows = padded.unfold(2, 2 * max_support + 1, 1)

        center = windows[..., max_support]
        left = windows[..., :max_support].flip(-1)
        right = windows[..., max_support + 1 :]
        left_ok = torch.cumprod((center.unsqueeze(-1) > left).to(x.dtype), dim=-1) > 0
        right_ok = torch.cumprod((center.unsqueeze(-1) > right).to(x.dtype), dim=-1) > 0
        peaks_by_support = left_ok & right_ok

        n_outputs = supports.shape[0]
        gather_indices = (
            (supports - 1).view(1, 1, 1, n_outputs).expand(batch_size, n_states, length, n_outputs)
        )
        peaks = torch.gather(peaks_by_support, dim=3, index=gather_indices)

        positions = torch.arange(length, device=x.device).view(1, 1, length, 1)
        support_row = supports.view(1, 1, 1, n_outputs)
        interior = (positions >= support_row) & (positions < length - support_row)
        return (peaks & interior).sum(dim=2).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format: ``number_peaks__n_<n>``.
        """
        return [f"number_peaks__n_{param['n']}" for param in self.params]


@torch.no_grad()
def find_peak_mask(x: Tensor, n: int = 1) -> Tensor:
    """Find strict local maxima with support ``n`` on both sides.

    Args:
        x: Tensor of shape (B, S, T).
        n: Number of neighbours tested on each side.

    Returns:
        Boolean tensor of shape (B, S, T).
    """
    batch_size, n_states, length = x.shape
    if n <= 0:
        raise ValueError("find_peak_mask requires a positive `n`.")
    if 2 * n >= length:
        return torch.zeros(batch_size, n_states, length, dtype=torch.bool, device=x.device)

    windows = x.unfold(2, 2 * n + 1, 1)
    center = windows[..., n]
    is_peak = (center > windows[..., :n].amax(dim=-1)) & (
        center > windows[..., n + 1 :].amax(dim=-1)
    )
    return functional.pad(is_peak, (n, n), value=False)


@torch.no_grad()
def extract_peak_values(x: Tensor, n: int = 1) -> tuple[Tensor, Tensor]:
    """Extract strict peak values for orbit-diagram construction.

    Args:
        x: Tensor of shape (B, S, T).
        n: Number of neighbours tested on each side.

    Returns:
        A tuple containing peak values of shape (max_peaks, B, S), padded with
        NaN, and peak counts of shape (B, S).
    """
    batch_size, n_states, _ = x.shape
    mask = find_peak_mask(x, n)
    peak_counts = mask.sum(dim=2)
    max_peaks = int(peak_counts.max().item()) if peak_counts.numel() else 0
    peak_values = torch.full(
        (max_peaks, batch_size, n_states),
        float("nan"),
        dtype=x.dtype,
        device=x.device,
    )
    if max_peaks == 0:
        return peak_values, peak_counts

    indices = mask.nonzero()
    positions = mask.long().cumsum(dim=2) - 1
    output_positions = positions[indices[:, 0], indices[:, 1], indices[:, 2]]
    peak_values[output_positions, indices[:, 0], indices[:, 1]] = x[
        indices[:, 0], indices[:, 1], indices[:, 2]
    ]
    return peak_values, peak_counts
