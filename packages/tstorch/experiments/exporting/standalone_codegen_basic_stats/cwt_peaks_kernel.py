# pyright: basic
from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional


def relative_maxima(x: Tensor) -> Tensor:
    """
    Find strict local maxima along the final dimension.

    :param x: Values whose final dimension is time.
    :return: A Boolean tensor with the same shape as ``x``.
    """
    interior = (x[..., 1:-1] > x[..., :-2]) & (x[..., 1:-1] > x[..., 2:])
    return functional.pad(interior, (1, 1), value=False)


def ricker_wavelet(points: int, width: int, *, dtype: torch.dtype = torch.float64) -> Tensor:
    """
    Construct SciPy's normalized Ricker wavelet.

    :param points: Number of samples in the wavelet.
    :param width: Wavelet width.
    :param dtype: Floating-point dtype for the returned tensor.
    :return: One-dimensional wavelet tensor.
    """
    width_tensor = torch.tensor(float(width), dtype=dtype)
    positions = torch.arange(points, dtype=dtype) - (points - 1.0) / 2.0
    coefficient = 2.0 / (torch.sqrt(3.0 * width_tensor) * math.pi**0.25)
    squared = positions.square()
    return (
        coefficient
        * (1.0 - squared / width_tensor.square())
        * torch.exp(-squared / (2.0 * width_tensor.square()))
    )


def continuous_wavelet_transform(x: Tensor, kernel_bank: Tensor) -> Tensor:
    """
    Apply fixed Ricker kernels with one batched ``conv1d``.

    :param x: Input tensor of shape ``(rows, time)``.
    :param kernel_bank: Zero-padded wavelets of shape ``(widths, kernel_length)``.
    :return: CWT tensor of shape ``(rows, widths, time)``.
    """
    kernel_length = kernel_bank.shape[1]
    signal = functional.pad(x.unsqueeze(1), (kernel_length // 2, (kernel_length - 1) // 2))
    return functional.conv1d(signal, kernel_bank.unsqueeze(1))


def direct_continuous_wavelet_transform(x: Tensor, kernel_bank: Tensor) -> Tensor:
    """
    Apply fixed Ricker kernels through ordered tap accumulation.

    :param x: Input tensor of shape ``(rows, time)``.
    :param kernel_bank: Zero-padded wavelets of shape ``(widths, kernel_length)``.
    :return: CWT tensor of shape ``(rows, widths, time)``.
    """
    length = x.shape[1]
    width_count, kernel_length = kernel_bank.shape
    padded = functional.pad(x, (kernel_length // 2, (kernel_length - 1) // 2))
    result = torch.zeros((x.shape[0], width_count, length), dtype=x.dtype, device=x.device)
    for tap in range(kernel_length):
        signal_at_tap = padded[:, tap : tap + length].unsqueeze(1)
        weights_at_tap = kernel_bank[:, tap].view(1, width_count, 1)
        result = result + signal_at_tap * weights_at_tap
    return result


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
    batch = row.shape[0]
    lower_values = torch.gather(ordered, 2, lower_rank.view(1, -1, 1).expand(batch, -1, 1)).squeeze(
        2
    )
    upper_values = torch.gather(ordered, 2, upper_rank.view(1, -1, 1).expand(batch, -1, 1)).squeeze(
        2
    )
    weight = upper_weight.unsqueeze(0)
    return lower_values + (upper_values - lower_values) * weight


def count_cwt_ridges(
    cwt: Tensor,
    max_distances: Tensor,
    *,
    gap_threshold: int,
    minimum_length: int,
    window_indices: Tensor,
    window_valid: Tensor,
    lower_rank: Tensor,
    upper_rank: Tensor,
    upper_weight: Tensor,
) -> Tensor:
    """
    Count SciPy-compatible CWT ridge lines with fixed-size tensor state.

    The ridge store has one slot for every possible ``(width, time)`` maximum.
    Ridge creation is data-dependent, but storage shapes and loop bounds depend
    only on the configured width count and input length.

    :param cwt: Wavelet coefficients of shape ``(rows, widths, time)``.
    :param max_distances: Maximum ridge displacement for every width.
    :param gap_threshold: Number of missing widths tolerated by a ridge.
    :param minimum_length: Minimum number of points in an accepted ridge.
    :param window_indices: Static indices for every local noise window.
    :param window_valid: Valid entries in ``window_indices``.
    :param lower_rank: Lower percentile rank for every noise window.
    :param upper_rank: Upper percentile rank for every noise window.
    :param upper_weight: Linear interpolation weight for every noise window.
    :return: Peak counts with shape ``(rows,)``.
    """
    n_rows, width_count, length = cwt.shape
    slot_count = width_count * length
    slot_ids = torch.arange(slot_count, device=cwt.device).view(1, slot_count)
    positions = torch.arange(length, device=cwt.device).view(1, length)
    sentinel = slot_count

    maxima = relative_maxima(cwt)
    created = torch.zeros((n_rows, slot_count), dtype=torch.bool, device=cwt.device)
    alive = torch.zeros_like(created)
    endpoints = torch.zeros((n_rows, slot_count), dtype=torch.long, device=cwt.device)
    gaps = torch.zeros_like(endpoints)
    ridge_lengths = torch.zeros_like(endpoints)
    first_rows = torch.zeros_like(endpoints)
    first_cols = torch.zeros_like(endpoints)
    next_slot = torch.zeros((n_rows, 1), dtype=torch.long, device=cwt.device)

    for row_index in range(width_count - 1, -1, -1):
        active_sources = torch.where(alive, slot_ids, torch.full_like(slot_ids, sentinel))
        active_endpoints = torch.where(alive, endpoints, torch.zeros_like(endpoints))
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
            left_positions >= 0, positions - left_positions, torch.full_like(positions, length + 1)
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

        row_maxima = maxima[:, row_index]
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
        alive = alive & (gaps <= gap_threshold)

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
    flattened = cwt.reshape(-1)
    coefficient_indices = row_offsets + first_rows * length + first_cols
    signal = torch.gather(flattened, 0, coefficient_indices.reshape(-1)).view_as(first_rows)
    noise = torch.gather(noises, 1, first_cols)
    snr = torch.abs(signal / noise)
    accepted = created & (ridge_lengths >= minimum_length) & ~(snr < 1.0)
    return accepted.sum(dim=1)


class NumberCwtPeaksKernel(nn.Module):
    """
    Shape-specialized, tensor-only implementation of ``number_cwt_peaks``.

    :param length: Static time dimension accepted by the kernel.
    :param width_counts: Requested inclusive width counts.
    :param convolution: CWT implementation selected at specialization time.

    :ivar length: Required time dimension.
    :ivar width_counts: Requested inclusive width counts.
    :ivar max_width_count: Number of CWT rows shared by all outputs.
    :ivar convolution: CWT implementation used by ``forward``.
    """

    wavelet_bank: Tensor
    max_distances: Tensor
    window_indices: Tensor
    window_valid: Tensor
    lower_rank: Tensor
    upper_rank: Tensor
    upper_weight: Tensor

    def __init__(
        self,
        *,
        length: int,
        width_counts: tuple[int, ...] = (1, 5),
        convolution: Literal["conv1d", "direct"] = "conv1d",
    ) -> None:
        super().__init__()
        if length < 3:
            raise ValueError("length must be at least three")
        if not width_counts or any(width_count < 1 for width_count in width_counts):
            raise ValueError("width_counts must contain positive integers")
        if convolution not in {"conv1d", "direct"}:
            raise ValueError("convolution must be 'conv1d' or 'direct'")
        self.length: int = length
        self.width_counts: tuple[int, ...] = width_counts
        self.max_width_count: int = max(width_counts)
        self.convolution: Literal["conv1d", "direct"] = convolution

        kernels = []
        for width in range(1, self.max_width_count + 1):
            points = min(10 * width, length)
            kernels.append(ricker_wavelet(points, width))
        kernel_length = max(kernel.shape[0] for kernel in kernels)
        wavelet_bank = torch.zeros((self.max_width_count, kernel_length), dtype=torch.float64)
        global_left_padding = kernel_length // 2
        for index, kernel in enumerate(kernels):
            offset = global_left_padding - kernel.shape[0] // 2
            wavelet_bank[index, offset : offset + kernel.shape[0]] = kernel
        self.register_buffer("wavelet_bank", wavelet_bank)

        self.register_buffer(
            "max_distances",
            torch.arange(1, self.max_width_count + 1, dtype=torch.float64) / 4.0,
        )
        window_size = math.ceil(length / 20)
        half_window, odd = divmod(window_size, 2)
        starts = [max(index - half_window, 0) for index in range(length)]
        ends = [min(index + half_window + odd, length) for index in range(length)]
        sizes = [end - start for start, end in zip(starts, ends, strict=True)]
        max_size = max(sizes)
        indices = torch.zeros((length, max_size), dtype=torch.long)
        valid = torch.zeros((length, max_size), dtype=torch.bool)
        for index, (start, size) in enumerate(zip(starts, sizes, strict=True)):
            indices[index, :size] = torch.arange(start, start + size)
            valid[index, :size] = True
        percentile_positions = torch.tensor(sizes, dtype=torch.float64).sub(1).mul(0.1)
        lower_rank = percentile_positions.floor().to(torch.long)
        upper_rank = percentile_positions.ceil().to(torch.long)
        self.register_buffer("window_indices", indices)
        self.register_buffer("window_valid", valid)
        self.register_buffer("lower_rank", lower_rank)
        self.register_buffer("upper_rank", upper_rank)
        self.register_buffer("upper_weight", percentile_positions - lower_rank)

    def forward(self, x: Tensor) -> Tensor:
        """
        Count CWT peaks in each input series.

        :param x: Tensor of shape ``(batch, states, self.length)``.
        :return: Tensor of shape ``(batch, states, len(self.width_counts))``.
        """
        torch._assert(x.shape[2] == self.length, "unexpected time dimension")
        batch, states, _ = x.shape
        rows = x.reshape(batch * states, self.length)
        kernel_bank = self.wavelet_bank.to(dtype=x.dtype)
        if self.convolution == "conv1d":
            cwt = continuous_wavelet_transform(rows, kernel_bank)
        else:
            cwt = direct_continuous_wavelet_transform(rows, kernel_bank)
        outputs = [
            count_cwt_ridges(
                cwt[:, :width_count],
                self.max_distances[:width_count],
                gap_threshold=1,
                minimum_length=math.ceil(width_count / 4),
                window_indices=self.window_indices,
                window_valid=self.window_valid,
                lower_rank=self.lower_rank,
                upper_rank=self.upper_rank,
                upper_weight=self.upper_weight,
            )
            for width_count in self.width_counts
        ]
        return torch.stack(outputs, dim=1).to(x.dtype).view(batch, states, len(self.width_counts))
