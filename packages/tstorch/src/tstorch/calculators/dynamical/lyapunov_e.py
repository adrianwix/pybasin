# pyright: basic
"""PyTorch implementation of Eckmann algorithm for multiple Lyapunov exponents.

This module provides GPU-accelerated computation of multiple Lyapunov exponents
using the Eckmann algorithm, vectorized for batch processing.

Reference:
    J. P. Eckmann, S. O. Kamphorst, D. Ruelle, and S. Ciliberto,
    "Liapunov exponents from time series," Physical Review A,
    vol. 34, no. 6, pp. 4971-4979, 1986.
"""

import torch
from torch import Tensor

from tstorch.torch_feature_utilities import delay_embedding


@torch.no_grad()
def lyap_e_single(
    data: Tensor,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> Tensor:
    """Compute multiple Lyapunov exponents for a single 1D time series.

    :param data: 1D time series tensor of shape (T,) where T is the number of time points.
    :param emb_dim: Embedding dimension (must satisfy (emb_dim-1) % (matrix_dim-1) == 0).
        Determines the phase space reconstruction dimension. Default is 10.
    :param matrix_dim: Matrix dimension for Jacobian estimation (number of exponents to compute).
        Default is 4.
    :param min_nb: Minimal number of neighbors required for reliable estimation. Default is 8.
    :param min_tsep: Minimal temporal separation between neighbors (to avoid spurious correlations).
        Default is 0.
    :param tau: Time step size for normalization of the exponents. Default is 1.0.
    :return: Tensor of shape (matrix_dim,) containing the Lyapunov exponents sorted from largest
        to smallest. Returns inf values for unbounded/divergent trajectories or when computation fails.
    """
    finite_data = torch.all(torch.isfinite(data))

    m = (emb_dim - 1) // (matrix_dim - 1)

    # Build orbit (delay embedding)
    orbit = delay_embedding(data[:-m], emb_dim, lag=1)
    n_orbit = len(orbit)

    old_q = torch.eye(matrix_dim, device=data.device, dtype=data.dtype)
    lexp = torch.zeros(matrix_dim, device=data.device, dtype=data.dtype)
    lexp_counts = torch.zeros(matrix_dim, device=data.device, dtype=data.dtype)

    # Pre-compute stepped data
    step_offsets = torch.arange(matrix_dim, device=data.device) * m
    stepped_data = data[
        torch.arange(n_orbit, device=data.device).unsqueeze(1) + step_offsets.unsqueeze(0)
    ]
    beta_data = data[torch.arange(n_orbit, device=data.device) + matrix_dim * m]

    all_indices = torch.arange(n_orbit, device=data.device)
    shift_upper = torch.cat(
        [
            torch.zeros((matrix_dim - 1, 1), device=data.device, dtype=data.dtype),
            torch.eye(matrix_dim - 1, device=data.device, dtype=data.dtype),
        ],
        dim=1,
    )

    for i in range(n_orbit):
        # Compute Chebyshev distances
        diffs = torch.max(torch.abs(orbit - orbit[i]), dim=1).values
        self_mask = all_indices == i
        diffs = torch.where(self_mask, torch.tensor(float("inf"), device=data.device), diffs)

        # Temporal masking
        time_mask = torch.abs(all_indices - i) <= min_tsep
        diffs = torch.where(time_mask, torch.tensor(float("inf"), device=data.device), diffs)

        # Find neighbors
        sorted_diffs, _ = torch.sort(diffs)
        r = sorted_diffs[min_nb - 1]

        neighbor_mask = diffs <= r
        n_neighbors = neighbor_mask.sum()
        valid_step = torch.isfinite(r) & (n_neighbors >= min_nb)

        # Build least squares matrices
        orbit_i_stepped = stepped_data[i]
        beta_i = beta_data[i]

        x_diff = stepped_data - orbit_i_stepped
        beta_diff = beta_data - beta_i

        weights = torch.where(
            valid_step,
            neighbor_mask.to(data.dtype),
            torch.zeros_like(neighbor_mask, dtype=data.dtype),
        )
        x_weighted = x_diff * weights.unsqueeze(1)

        # Solve least squares using normal equations
        xtx = x_weighted.T @ x_diff + 1e-10 * torch.eye(
            matrix_dim, device=data.device, dtype=data.dtype
        )
        xty = x_weighted.T @ beta_diff
        a = torch.linalg.pinv(xtx) @ xty
        a = torch.where(valid_step, a, torch.zeros_like(a))

        # Build T matrix
        t_mat = torch.cat([shift_upper, a.unsqueeze(0)], dim=0)

        # QR decomposition
        mat_q, mat_r = torch.linalg.qr(t_mat @ old_q)

        # Force positive diagonal
        sign_diag = torch.sign(torch.diag(mat_r))
        sign_diag = torch.where(sign_diag == 0, torch.ones_like(sign_diag), sign_diag)
        mat_q = mat_q @ torch.diag(sign_diag)
        mat_r = torch.diag(sign_diag) @ mat_r

        old_q = torch.where(valid_step, mat_q, old_q)

        # Accumulate Lyapunov exponents
        diag_r = torch.diag(mat_r)
        pos_mask = (diag_r > 0) & valid_step
        lexp = lexp + torch.where(pos_mask, torch.log(diag_r), torch.zeros_like(diag_r))
        lexp_counts = lexp_counts + pos_mask.to(data.dtype)

    # Normalize
    valid_mask = lexp_counts > 0
    lexp = torch.where(valid_mask, lexp / torch.clamp_min(lexp_counts, 1), float("inf"))
    lexp /= tau
    lexp /= m

    return torch.where(
        finite_data,
        lexp,
        torch.full((matrix_dim,), float("inf"), dtype=data.dtype, device=data.device),
    )
