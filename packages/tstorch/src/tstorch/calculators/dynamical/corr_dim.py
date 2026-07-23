# pyright: basic
"""PyTorch implementation of Grassberger-Procaccia algorithm for correlation dimension.

This module provides GPU-accelerated computation of correlation dimension
using the Grassberger-Procaccia algorithm, vectorized for batch processing.

Reference:
    P. Grassberger and I. Procaccia, "Characterization of strange
    attractors," Physical review letters, vol. 50, no. 5, p. 346, 1983.
"""

import torch
from torch import Tensor

from tstorch.torch_feature_utilities import delay_embedding


def _logarithmic_r(min_r: Tensor, max_r: Tensor, n_rvals: int, device: torch.device) -> Tensor:
    """Create logarithmically spaced radius values (vmap-compatible).

    :param min_r: Minimum radius value as a scalar tensor.
    :param max_r: Maximum radius value as a scalar tensor.
    :param n_rvals: Number of logarithmically spaced values to generate.
    :param device: Device to create the output tensor on (e.g., 'cpu' or 'cuda').
    :return: Tensor of shape (n_rvals,) containing logarithmically spaced values
        from min_r to max_r.
    """
    log_min = torch.log(min_r + 1e-10)
    log_max = torch.log(max_r + 1e-10)
    # Create interpolation weights [0, 1]
    alphas = torch.linspace(0, 1, n_rvals, device=device)
    # Interpolate in log space: log_min + alpha * (log_max - log_min)
    log_vals = log_min + alphas * (log_max - log_min)
    return torch.exp(log_vals)


@torch.no_grad()
def corr_dim_single(
    data: Tensor,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Tensor:
    """Compute correlation dimension for a single 1D time series.

    This is the Grassberger-Procaccia algorithm. The correlation dimension
    is a characteristic measure that describes the geometry of chaotic attractors.
    It quantifies how the correlation sum scales with radius in the reconstructed
    phase space.

    :param data: 1D time series tensor of shape (T,) where T is the number of time points.
    :param emb_dim: Embedding dimension for phase space reconstruction. Determines the
        dimensionality of the reconstructed attractor. Default is 4.
    :param lag: Lag (delay) for delay embedding. Specifies the time delay between
        consecutive elements in the embedding vectors. Default is 1.
    :param n_rvals: Number of radius values to use in the correlation integral computation.
        More values provide better resolution but increase computation time. Default is 50.
    :return: Correlation dimension as a scalar tensor. This represents the slope of log(C)
        vs log(r) where C is the correlation sum and r is the radius. Returns NaN if
        computation fails (e.g., insufficient data or all infinite distances).
    """
    # Create delay embedding
    orbit = delay_embedding(data, emb_dim, lag)
    n = orbit.shape[0]

    # Compute pairwise Euclidean distances
    dists = torch.cdist(orbit, orbit, p=2)

    # Zero out diagonal (self-distances)
    mask = torch.eye(n, dtype=torch.bool, device=data.device)
    dists = dists.masked_fill(mask, float("inf"))

    # Compute rvals based on data statistics
    sd = torch.std(data, correction=1)
    min_r = 0.1 * sd
    max_r = 0.5 * sd
    rvals = _logarithmic_r(min_r, max_r, n_rvals, data.device)

    # Vectorized: compare all distances against all radii at once
    csums = (dists.unsqueeze(0) <= rvals.view(-1, 1, 1)).sum(dim=(1, 2)) / (n * (n - 1))

    # Filter out zero csums and fit line in log-log space
    nonzero_mask = csums > 0
    log_r = torch.log(rvals)
    log_c = torch.where(
        nonzero_mask,
        torch.log(csums + 1e-30),
        torch.tensor(float("nan"), device=data.device),
    )

    # Linear regression: log(C) = D * log(r) + intercept
    n_valid = nonzero_mask.sum()

    log_r_valid = torch.where(nonzero_mask, log_r, torch.zeros_like(log_r))
    log_c_valid = torch.where(nonzero_mask, log_c, torch.zeros_like(log_c))

    sum_r = log_r_valid.sum()
    sum_c = log_c_valid.sum()
    sum_rr = (log_r_valid * log_r_valid).sum()
    sum_rc = (log_r_valid * log_c_valid).sum()

    denom = n_valid * sum_rr - sum_r * sum_r
    slope = torch.where(
        (n_valid >= 2) & (torch.abs(denom) > 1e-10),
        (n_valid * sum_rc - sum_r * sum_c) / denom,
        torch.tensor(float("nan"), device=data.device),
    )

    return slope
