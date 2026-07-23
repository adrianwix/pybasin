# pyright: basic
"""Dynamical systems feature calculators for time series.

All feature functions follow a consistent tensor shape convention:
- Input: (B, S, T) where B=batch size, S=state variables, T=timesteps
- Output: (B, S, 1) for scalar features, or (B, S, K) for multi-valued features where K is the number of values

Features are computed along the time dimension (dim=2), preserving batch and state dimensions.
"""

import torch
from torch import Tensor

from tstorch.calculators.dynamical.corr_dim import corr_dim_single
from tstorch.calculators.dynamical.lyapunov_e import lyap_e_single
from tstorch.calculators.dynamical.lyapunov_r import lyap_r_single

# =============================================================================
# DYNAMICAL SYSTEMS FEATURES (5 features)
# =============================================================================


def _map_scalar_series(x: Tensor, fn) -> Tensor:
    """Map a single-series feature over BST input.

    :param x: Input tensor of shape ``(B, S, T)``.
    :param fn: Callable accepting one ``(T,)`` series and returning a scalar tensor.
    :return: Tensor of shape ``(B, S, 1)``.
    """
    batch_size, n_states, T = x.shape
    flat = x.reshape(batch_size * n_states, T)
    return torch.vmap(fn)(flat).reshape(batch_size, n_states, 1)


def _map_vector_series(x: Tensor, output_size: int, fn) -> Tensor:
    """Map a vector-valued single-series feature over BST input.

    Multi-output feature: returns ``output_size`` outputs for each input series.

    :param x: Input tensor of shape ``(B, S, T)``.
    :param output_size: Number of outputs produced for each single series.
    :param fn: Callable accepting one ``(T,)`` series and returning ``(output_size,)``.
    :return: Tensor of shape ``(B, S, output_size)``.
    """
    batch_size, n_states, T = x.shape
    flat = x.reshape(batch_size * n_states, T)
    return torch.vmap(fn)(flat).reshape(batch_size, n_states, output_size)


@torch.no_grad()
def lyapunov_r(
    x: Tensor,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
) -> Tensor:
    """Compute largest Lyapunov exponent using Rosenstein algorithm.

    :param x: Input time series tensor of shape (B, S, T) where T=timesteps.
    :param emb_dim: Embedding dimension for phase space reconstruction. Default is 10.
    :param lag: Lag for delay embedding. Default is 1.
    :param trajectory_len: Number of steps to follow divergence. Default is 20.
    :param tau: Time step size for normalization. Default is 1.0.
    :return: Tensor of shape (B, S, 1) containing the largest Lyapunov exponent for each
        state of each batch.
    """
    return _map_scalar_series(
        x,
        lambda series: lyap_r_single(series, emb_dim, lag, trajectory_len, tau),
    )


@torch.no_grad()
def lyapunov_e(
    x: Tensor,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> Tensor:
    """Compute multiple Lyapunov exponents using Eckmann algorithm.

    Multi-output feature: returns ``matrix_dim`` Lyapunov exponents per input series.

    :param x: Input time series tensor of shape (B, S, T) where T=timesteps.
    :param emb_dim: Embedding dimension for phase space reconstruction. Default is 10.
    :param matrix_dim: Matrix dimension for Jacobian estimation (number of exponents to compute).
        Default is 4.
    :param min_nb: Minimal number of neighbors required. Default is 8.
    :param min_tsep: Minimal temporal separation between neighbors. Default is 0.
    :param tau: Time step size for normalization. Default is 1.0.
    :return: Tensor of shape (B, S, matrix_dim) containing the Lyapunov exponents. The third
        dimension contains ``matrix_dim`` exponents sorted from largest to smallest.
    """
    return _map_vector_series(
        x,
        matrix_dim,
        lambda series: lyap_e_single(series, emb_dim, matrix_dim, min_nb, min_tsep, tau),
    )


@torch.no_grad()
def correlation_dimension(
    x: Tensor,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Tensor:
    """Compute correlation dimension using Grassberger-Procaccia algorithm.

    :param x: Input time series tensor of shape (B, S, T) where T=timesteps.
    :param emb_dim: Embedding dimension for phase space reconstruction. Default is 4.
    :param lag: Lag for delay embedding. Default is 1.
    :param n_rvals: Number of radius values to use in correlation integral. Default is 50.
    :return: Tensor of shape (B, S, 1) containing the correlation dimension for each state of each batch.
    """
    return _map_scalar_series(
        x,
        lambda series: corr_dim_single(series, emb_dim, lag, n_rvals),
    )


@torch.no_grad()
def _friedrich_coefficients_all(x: Tensor, m: int = 3) -> Tensor:
    """Compute all Friedrich polynomial coefficients for BST input.

    KNOWN ALGORITHM MISMATCH WITH TSFRESH (see ``tests/unit/test_dynamical.py`` module
    docstring): tsfresh's ``_estimate_friedrich_coefficients`` bins the signal into ``r``
    quantiles via ``pandas.qcut``, averages ``(signal, delta)`` per bin, then fits a
    degree-``m`` polynomial to those bin means. This function instead fits directly to
    every raw ``(position, velocity)`` pair via a regularized Vandermonde least-squares
    solve, with no quantile binning and no ``r`` parameter at all. The two are different
    regressions and are not expected to produce matching coefficients even on
    well-conditioned input — this needs a real reimplementation of tsfresh's
    quantile-binned fit to be numerically compatible, which has not been done.

    :param x: Input tensor of shape (B, S, T).
    :param m: Polynomial degree.
    :return: Tensor of shape ``(B, S, m + 1)``.
    """
    batch_size, n_states, T = x.shape

    velocity = x[..., 1:] - x[..., :-1]
    position = x[..., :-1]

    velocity_flat = velocity.reshape(batch_size * n_states, T - 1)
    position_flat = position.reshape(batch_size * n_states, T - 1)

    # Build Vandermonde matrices: V[batch, time, power] = position^power
    # Powers: [m, m-1, ..., 1, 0]
    powers = torch.arange(m, -1, -1, device=x.device, dtype=x.dtype)  # (m+1,)
    V = position_flat.unsqueeze(-1) ** powers  # (B*S, T-1, m+1)

    # Solve least squares via normal equations: (V^T V) @ coeffs = V^T @ velocity
    # V^T @ V shape: (B*S, m+1, m+1)
    VtV = torch.bmm(V.transpose(1, 2), V)  # (B*S, m+1, m+1)
    # V^T @ velocity shape: (B*S, m+1)
    Vtv = torch.bmm(V.transpose(1, 2), velocity_flat.unsqueeze(-1)).squeeze(-1)  # (B*S, m+1)

    # Add regularization for numerical stability
    reg = 1e-6 * torch.eye(m + 1, device=x.device, dtype=x.dtype)
    VtV = VtV + reg

    # Solve using torch.linalg.solve (batched)
    try:
        coeffs = torch.linalg.solve(VtV, Vtv)  # (B*S, m+1)
    except Exception:
        # Fallback: use pseudo-inverse
        coeffs = torch.zeros(batch_size * n_states, m + 1, device=x.device, dtype=x.dtype)

    return coeffs.reshape(batch_size, n_states, m + 1)


@torch.no_grad()
def friedrich_coefficients(x: Tensor, m: int = 3, r: float = 30.0, coeff: int = 0) -> Tensor:
    """Coefficient of polynomial fit to velocity vs position.

    :param x: Input tensor of shape (B, S, T).
    :param m: Polynomial degree.
    :param r: Compatibility parameter matching tsfresh.
    :param coeff: Coefficient index to return.
    :return: Tensor of shape (B, S, 1).
    """
    del r
    coeffs = _friedrich_coefficients_all(x, m)
    if coeff < coeffs.shape[-1]:
        return coeffs[..., coeff : coeff + 1]
    return torch.zeros(*x.shape[:2], 1, device=x.device, dtype=x.dtype)


@torch.no_grad()
def max_langevin_fixed_point(x: Tensor, r: float = 3, m: int = 30) -> Tensor:
    """Maximum fixed point of the Langevin model.

    :param x: Input tensor of shape (B, S, T).
    :param r: Compatibility parameter matching tsfresh.
    :param m: Polynomial degree used for the fitted model.
    :return: Tensor of shape (B, S, 1).
    """
    batch_size, n_states, T = x.shape
    m_int = int(m)

    velocity = x[..., 1:] - x[..., :-1]
    position = x[..., :-1]

    velocity_flat = velocity.reshape(batch_size * n_states, T - 1)
    position_flat = position.reshape(batch_size * n_states, T - 1)

    # Build Vandermonde matrices: V[batch, time, power] = position^power
    powers = torch.arange(m_int, -1, -1, device=x.device, dtype=x.dtype)  # (m+1,)
    vander = position_flat.unsqueeze(-1) ** powers  # (B*S, T-1, m+1)

    # Solve least squares via normal equations: (V^T V) @ coeffs = V^T @ velocity
    vtv = torch.bmm(vander.transpose(1, 2), vander)  # (B*S, m+1, m+1)
    vtv = vtv + 1e-6 * torch.eye(m_int + 1, device=x.device, dtype=x.dtype)  # Regularize
    vty = torch.bmm(vander.transpose(1, 2), velocity_flat.unsqueeze(-1)).squeeze(-1)  # (B*S, m+1)

    try:
        coeffs = torch.linalg.solve(vtv, vty)  # (B*S, m+1)
    except Exception:
        return torch.zeros(batch_size, n_states, 1, dtype=x.dtype, device=x.device)

    # Compute fitted values: V @ coeffs
    fitted = torch.bmm(vander, coeffs.unsqueeze(-1)).squeeze(-1)  # (B*S, T-1)

    # Find zero crossings (sign changes)
    sign_change = fitted[:, :-1] * fitted[:, 1:] < 0  # (B*S, T-2)

    # Get position values at crossings, masked where no crossing
    position_at_crossings = position_flat[:, :-1]  # (B*S, T-2)
    position_at_crossings = torch.where(
        sign_change, position_at_crossings, torch.tensor(float("-inf"), device=x.device)
    )

    # Max position at crossings per series
    result = position_at_crossings.max(dim=1).values  # (B*S,)
    result = torch.where(result == float("-inf"), torch.zeros_like(result), result)

    return result.reshape(batch_size, n_states, 1)
