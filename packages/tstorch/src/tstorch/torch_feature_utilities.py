# pyright: basic

"""PyTorch utility functions for feature extraction.

This module provides utility functions for handling NaN, inf, and other
edge cases in feature arrays, similar to tsfresh's dataframe_functions.
"""

import torch
from torch import Tensor


def welch_psd(x: Tensor, *, scaled: bool = True) -> Tensor:
    """Welch-method power spectral density, matching ``scipy.signal.welch`` defaults.

    Reproduces ``welch(x, nperseg=min(len(x), 256))``: segments of length ``nperseg``
    with 50% overlap (``noverlap = nperseg // 2``), per-segment mean removal
    (``detrend='constant'``), a periodic Hann window, one-sided FFT with the standard
    non-DC/non-Nyquist doubling, averaged across segments. Shared by any calculator that
    needs tsfresh's Welch PSD (``fourier_entropy``, ``spkt_welch_density``) instead of
    reimplementing it per file.

    :param x: Input tensor of shape (B, S, T).
    :param scaled: If ``True`` (default), apply the ``1 / (fs * sum(window**2))``
        density-scaling factor scipy's ``welch`` uses, matching its absolute PSD values.
        Pass ``False`` when only the PSD's *shape* matters (e.g. as an input to
        max-normalized binned entropy), where that factor is a per-series constant that
        cancels out anyway and skipping it saves a division.
    :return: Tensor of shape (B, S, nperseg // 2 + 1).
    """
    batch_size, n_states, T = x.shape
    nperseg = min(T, 256)
    noverlap = nperseg // 2
    step = nperseg - noverlap

    flat = x.reshape(batch_size * n_states, T)
    segments = flat.unfold(dimension=1, size=nperseg, step=step)  # (B*S, n_segments, nperseg)
    segments = segments - segments.mean(dim=-1, keepdim=True)

    window = torch.hann_window(nperseg, periodic=True, dtype=x.dtype, device=x.device)
    segments = segments * window

    fft_result = torch.fft.rfft(segments, dim=-1)
    power = fft_result.real**2 + fft_result.imag**2
    if nperseg % 2 == 0:
        power[..., 1:-1] *= 2
    else:
        power[..., 1:] *= 2

    if scaled:
        power = power / (window**2).sum()  # fs=1.0, so scale == 1 / sum(window**2)

    pxx = power.mean(dim=1)
    return pxx.reshape(batch_size, n_states, -1)


def impute(features: Tensor) -> Tensor:
    """
    Columnwise replaces all NaNs and infs from the feature array with average/extreme values.

    This is done as follows for each column:
        * -inf -> min (of finite values in that column)
        * +inf -> max (of finite values in that column)
        * NaN -> median (of finite values in that column)

    If a column does not contain any finite values at all, it is filled with zeros.

    This function is the PyTorch equivalent of tsfresh's impute function.

    :param features: Feature tensor of shape (B, F) where:
        - B: batch size (number of samples/trajectories)
        - F: number of features
    :return: Imputed feature tensor of shape (B, F), guaranteed to contain no NaN or inf values.
        Each column is imputed independently using the strategy described above.
    """
    result = features.clone()
    n_cols = features.shape[1]

    for col_idx in range(n_cols):
        col = features[:, col_idx]

        finite_mask = torch.isfinite(col)
        finite_vals = col[finite_mask]

        if finite_vals.numel() == 0:
            result[:, col_idx] = 0.0
            continue

        col_min = finite_vals.min()
        col_max = finite_vals.max()
        col_median = finite_vals.median()

        neg_inf_mask = torch.isneginf(col)
        pos_inf_mask = torch.isposinf(col)
        nan_mask = torch.isnan(col)

        result[neg_inf_mask, col_idx] = col_min
        result[pos_inf_mask, col_idx] = col_max
        result[nan_mask, col_idx] = col_median

    return result


def impute_extreme(features: Tensor, extreme_value: float = 1e10) -> Tensor:
    """
    Replaces all NaNs and infs with extreme values to make them distinguishable.

    This is useful when you want samples with non-finite features to be
    classified separately (e.g., unbounded trajectories in dynamical systems).

    - +inf -> +extreme_value
    - -inf -> -extreme_value
    - NaN -> +extreme_value (to cluster with +inf)

    :param features: Feature tensor of shape (B, F) where:
        - B: batch size (number of samples/trajectories)
        - F: number of features
    :param extreme_value: The extreme value to use for replacement. Default is 1e10.
    :return: Imputed feature tensor of shape (B, F). All inf and NaN values are replaced
        with ±extreme_value, allowing unbounded trajectories to be distinguished in clustering.
    """
    result = features.clone()
    result = torch.where(
        torch.isposinf(features),
        torch.tensor(extreme_value, device=features.device, dtype=features.dtype),
        result,
    )
    result = torch.where(
        torch.isneginf(features),
        torch.tensor(-extreme_value, device=features.device, dtype=features.dtype),
        result,
    )
    result = torch.where(
        torch.isnan(features),
        torch.tensor(extreme_value, device=features.device, dtype=features.dtype),
        result,
    )
    return result


def delay_embedding(data: Tensor, emb_dim: int, lag: int = 1) -> Tensor:
    """Create delay embedding of a time series.

    Constructs a matrix of delayed vectors from a 1D time series, useful for phase space
    reconstruction in dynamical systems analysis.

    :param data: Time series tensor. Can be:
        - 1D: shape (T,) where T is the number of time points
        - Higher dimensional: shape (..., T) where leading dimensions are preserved
    :param emb_dim: Embedding dimension. Determines how many delayed copies of the signal
        are concatenated to form each embedding vector.
    :param lag: Lag (delay) between elements in embedded vectors. Specifies the time delay
        in units of the sampling interval. Default is 1.
    :return: Embedded data tensor of shape (M, emb_dim) for 1D input, where
        M = T - (emb_dim-1)*lag is the number of embedded vectors. Each row contains
        [data[i], data[i+lag], data[i+2*lag], ..., data[i+(emb_dim-1)*lag]].
    """
    T = data.shape[-1]
    m = T - (emb_dim - 1) * lag
    indices = torch.arange(emb_dim, device=data.device) * lag
    indices = indices.unsqueeze(0) + torch.arange(m, device=data.device).unsqueeze(1)
    if data.ndim == 1:
        return data[indices]
    return data[..., indices]


def rowwise_euclidean(x: Tensor, y: Tensor) -> Tensor:
    """Compute Euclidean distance from each row of x to vector y.

    :param x: Matrix of shape (M, D) where M is the number of vectors and D is the dimension.
    :param y: Vector of shape (D,) to compute distances from.
    :return: Tensor of shape (M,) containing the Euclidean (L2) distance from each row
        of x to vector y.
    """
    return torch.norm(x - y, dim=-1)


def rowwise_chebyshev(x: Tensor, y: Tensor) -> Tensor:
    """Compute Chebyshev (L-infinity) distance from each row of x to vector y.

    The Chebyshev distance is the maximum absolute difference across all dimensions.

    :param x: Matrix of shape (M, D) where M is the number of vectors and D is the dimension.
    :param y: Vector of shape (D,) to compute distances from.
    :return: Tensor of shape (M,) containing the Chebyshev (L-infinity) distance from each
        row of x to vector y. Each distance is max(|x[i] - y|) across all dimensions.
    """
    return torch.max(torch.abs(x - y), dim=1).values


@torch.no_grad()
def local_maxima_1d(x: Tensor) -> Tensor:
    """Find local maxima in batched 1D signals.

    Equivalent to scipy's ``_local_maxima_1d``. Finds all local maxima (including
    plateaus) and returns a boolean mask with True at the midpoint of each maximum.

    A maximum is defined as one or more samples of equal value that are surrounded
    on both sides by at least one smaller sample. For plateaus, the midpoint
    (rounded down for even sizes) is marked as the peak.

    Vectorized via run-length encoding (same trick as
    ``torch_features_pattern.py``'s longest-strike fix - see
    ``.github/skills/tstorch-feature-perf/references/performance-notes.md``): every
    maximal run of equal consecutive values gets one id via a cumulative sum over
    change-points, then each run's start/end index is looked up with one
    ``scatter_reduce_`` (amin/amax) instead of scipy's per-signal, per-timestep
    sequential scan - no data-dependent Python branching, so this compiles under
    ``torch.compile(fullgraph=True)``.

    :param x: Input tensor of shape (B, S, T).
    :return: Boolean mask of shape (B, S, T) with True at peak positions.
        First and last samples are always False.
    """
    batch_size, n_states, T = x.shape
    device = x.device

    if T < 3:
        return torch.zeros(batch_size, n_states, T, dtype=torch.bool, device=device)

    changed = torch.cat(
        [
            torch.ones(batch_size, n_states, 1, dtype=torch.bool, device=device),
            x[..., 1:] != x[..., :-1],
        ],
        dim=-1,
    )
    group_id = changed.long().cumsum(dim=-1) - 1  # (B, S, T): run id per position

    idx = (
        torch.arange(T, device=device, dtype=torch.long)
        .view(1, 1, T)
        .expand(batch_size, n_states, T)
    )
    run_start = torch.full((batch_size, n_states, T), T, dtype=torch.long, device=device)
    run_start.scatter_reduce_(-1, group_id, idx, reduce="amin", include_self=True)
    run_end = torch.full((batch_size, n_states, T), -1, dtype=torch.long, device=device)
    run_end.scatter_reduce_(-1, group_id, idx, reduce="amax", include_self=True)

    pos_run_start = run_start.gather(-1, group_id)  # this position's run's first index
    pos_run_end = run_end.gather(-1, group_id)  # this position's run's last index

    # A run is a candidate peak only if it doesn't touch either boundary (scipy only
    # scans positions 1..T-2); every position in the run shares its run's value (`x`
    # itself), so no extra gather is needed for the "signal[i]" comparand.
    interior = (pos_run_start >= 1) & (pos_run_end <= T - 2)
    left_val = x.gather(-1, (pos_run_start - 1).clamp(min=0))
    right_val = x.gather(-1, (pos_run_end + 1).clamp(max=T - 1))
    is_peak_run = interior & (left_val < x) & (right_val < x)

    midpoint = (pos_run_start + pos_run_end) // 2
    return is_peak_run & (idx == midpoint)
