# pyright: basic

"""PyTorch utility functions for feature extraction."""

import torch
from torch import Tensor


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
        torch
        .arange(T, device=device, dtype=torch.long)
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
