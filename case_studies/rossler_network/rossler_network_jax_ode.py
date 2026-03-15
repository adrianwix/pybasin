"""JAX-native Rössler network ODE system for basin stability of synchronization.

This module implements a network of N coupled Rössler oscillators with diffusive
coupling through the x-components via sparse Laplacian operations.
"""

from typing import TypedDict

import jax.numpy as jnp
from jax import Array

from pybasin.solvers.jax_ode_system import JaxODESystem


class RosslerNetworkParams(TypedDict):
    """Scalar ODE parameters for the coupled Rössler network."""

    a: float  # Rössler parameter a
    b: float  # Rössler parameter b
    c: float  # Rössler parameter c
    K: float  # Coupling strength


class RosslerNetworkJaxODE(JaxODESystem[RosslerNetworkParams]):
    """
        JAX-native coupled Rössler network ODE system with sparse Laplacian.

        The dynamics for each node i are:
            dx_i/dt = -y_i - z_i - K * sum_j(L_ij * x_j)
            dy_i/dt = x_i + a * y_i
            dz_i/dt = b + z_i * (x_i - c)

        Uses sparse Laplacian computation: O(E) instead of O(N²).

        The state vector y has shape (3*N,) organized as:
            [x_0, x_1, ..., x_{N-1}, y_0, y_1, ..., y_{N-1}, z_0, z_1, ..., z_{N-1}]

    :param params: Dictionary with keys ``a``, ``b``, ``c`` (Rössler oscillator parameters,
            typically ``a=b=0.2``, ``c=7.0``) and ``K`` (coupling strength).
    :param n: Number of nodes in the network.
    :param edges_i: Source node indices for the sparse Laplacian, shape ``(n_edges,)``.
    :param edges_j: Target node indices for the sparse Laplacian, shape ``(n_edges,)``.
    """

    def __init__(
        self, params: RosslerNetworkParams, n: int, edges_i: Array, edges_j: Array
    ) -> None:
        super().__init__(params)
        self._N = n
        self._edges_i = edges_i
        self._edges_j = edges_j

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        """
        Right-hand side of the coupled Rössler network.

        :param t: Current time, scalar.
        :param y: State vector of shape ``(3*N,)`` laid out as
            ``[x_0..x_{N-1}, y_0..y_{N-1}, z_0..z_{N-1}]``.
        :param p: Parameter array of shape ``(4,)`` ordered as ``[a, b, c, K]``.
        :return: Time derivatives of shape ``(3*N,)``.
        """
        a, b, c, k = p[0], p[1], p[2], p[3]
        n = self._N
        edges_i = self._edges_i
        edges_j = self._edges_j

        x = y[:n]
        y_state = y[n : 2 * n]
        z = y[2 * n :]

        diff = x[edges_i] - x[edges_j]
        coupling = jnp.zeros_like(x).at[edges_i].add(diff)  # pyright: ignore[reportUnknownMemberType]

        dx_dt = -y_state - z - k * coupling
        dy_dt = x + a * y_state
        dz_dt = b + z * (x - c)

        return jnp.concatenate([dx_dt, dy_dt, dz_dt])
