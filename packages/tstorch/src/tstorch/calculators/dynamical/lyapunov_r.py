# pyright: basic
"""PyTorch implementation of Rosenstein algorithm for largest Lyapunov exponent.

This module provides GPU-accelerated computation of the largest Lyapunov exponent
using the Rosenstein algorithm, vectorized for batch processing.

Reference:
    M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
    "A practical method for calculating largest Lyapunov exponents from
    small data sets," Physica D: Nonlinear Phenomena, vol. 65, no. 1,
    pp. 117-134, 1993.
"""

import torch
from torch import Tensor

from tstorch.torch_feature_utilities import delay_embedding


def compute_min_tsep(data: Tensor, T: int) -> Tensor:
    """Compute minimum temporal separation using mean frequency.

    Analyzes the power spectral density of the signal to determine an appropriate
    minimum temporal separation for neighbor selection.

    :param data: 1D time series tensor of shape (T,) where T is the number of time points.
    :param T: Length of data (typically data.shape[0]).
    :return: Minimum temporal separation as a scalar tensor. The value is clamped between
        1 and 25% of the signal length.
    """
    max_tsep_factor = 0.25

    f = torch.fft.rfft(data, n=T * 2 - 1)
    freqs = torch.fft.rfftfreq(T * 2 - 1, device=data.device)
    psd = torch.abs(f) ** 2

    mf = torch.sum(freqs[1:] * psd[1:]) / (torch.sum(psd[1:]) + 1e-10)
    min_tsep = torch.ceil(1.0 / (mf + 1e-10)).long()

    max_tsep = int(max_tsep_factor * T)
    min_tsep = torch.clamp(min_tsep, min=1, max=max_tsep)

    return min_tsep


def _poly_line_fit(ks: Tensor, div_traj: Tensor, finite_mask: Tensor) -> Tensor:
    """Fit a line using least squares.

    :param ks: x-values (k indices) tensor of shape (T,) where T is the number of data points.
    :param div_traj: y-values (divergence trajectory) tensor of shape (T,).
    :param finite_mask: Boolean mask for valid points of shape (T,). True indicates a valid point.
    :return: Slope of the fitted line as a scalar tensor. Returns NaN if insufficient valid points.
    """
    n_finite = finite_mask.sum()

    ks_finite = torch.where(finite_mask, ks, torch.zeros_like(ks))
    div_finite = torch.where(finite_mask, div_traj, torch.zeros_like(div_traj))

    sum_k = ks_finite.sum()
    sum_d = div_finite.sum()
    sum_kk = (ks_finite * ks_finite).sum()
    sum_kd = (ks_finite * div_finite).sum()

    denom = n_finite * sum_kk - sum_k * sum_k
    slope = torch.where(
        (n_finite >= 2) & (torch.abs(denom) > 1e-10),
        (n_finite * sum_kd - sum_k * sum_d) / denom,
        torch.tensor(float("nan"), device=ks.device),
    )

    return slope


@torch.no_grad()
def lyap_r_single(
    data: Tensor,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
) -> Tensor:
    """Compute largest Lyapunov exponent for a single 1D time series.

    Uses the Rosenstein algorithm to estimate the largest Lyapunov exponent by tracking
    the average logarithmic divergence of initially nearby trajectories in phase space.

    :param data: 1D time series tensor of shape (T,) where T is the number of time points.
    :param emb_dim: Embedding dimension for phase space reconstruction. Determines the
        dimensionality of the reconstructed attractor. Default is 10.
    :param lag: Lag (delay) for delay embedding. Specifies the time delay between
        consecutive elements in the embedding vectors. Default is 1.
    :param trajectory_len: Number of time steps to follow the divergence of nearby trajectories.
        Longer values provide more data for the linear fit but may be affected by saturation.
        Default is 20.
    :param tau: Time step size for normalization of the exponent. Default is 1.0.
    :return: Largest Lyapunov exponent as a scalar tensor. Returns NaN if computation fails
        (e.g., insufficient data points or invalid trajectory).
    """
    T = data.shape[0]
    min_tsep = compute_min_tsep(data, T)

    orbit = delay_embedding(data, emb_dim, lag)
    m = orbit.shape[0]

    # Compute pairwise distances
    dists = torch.cdist(orbit, orbit, p=2)

    # Mask self and temporally close neighbors.
    mask_range = torch.arange(m, device=data.device)
    for_masking = torch.abs(mask_range.unsqueeze(1) - mask_range.unsqueeze(0)) <= min_tsep
    dists = torch.where(for_masking, torch.tensor(float("inf"), device=data.device), dists)

    ntraj = max(m - trajectory_len + 1, 1)

    # Find nearest neighbors for each point
    search_dists = dists[:ntraj, :ntraj].clone()
    nb_idx = torch.argmin(search_dists, dim=1)

    # Compute divergence trajectory
    div_values: list[Tensor] = []
    for k in range(trajectory_len):
        i_indices = torch.arange(ntraj, device=data.device) + k
        j_indices = nb_idx + k

        valid_i = i_indices < m
        valid_j = j_indices < m
        valid = valid_i & valid_j

        div_k = torch.where(
            valid,
            dists[i_indices.clamp(max=m - 1), j_indices.clamp(max=m - 1)],
            torch.tensor(float("nan"), device=data.device),
        )

        div_k_safe = torch.where((div_k > 0) & valid, div_k, torch.ones_like(div_k))
        log_div_k = torch.log(div_k_safe)
        log_div_k = torch.where(
            (div_k > 0) & valid, log_div_k, torch.tensor(float("nan"), device=data.device)
        )

        div_values.append(torch.nanmean(log_div_k))

    div_traj = torch.stack(div_values)

    # Fit line to divergence trajectory
    ks = torch.arange(trajectory_len, dtype=torch.float32, device=data.device)
    finite_mask = torch.isfinite(div_traj)

    slope = _poly_line_fit(ks, div_traj, finite_mask)
    le = slope / tau

    return le
