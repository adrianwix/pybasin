"""JAX-native Rössler network ODE system for basin stability of synchronization.

This module implements a network of N coupled Rössler oscillators with diffusive
coupling through the x-components via sparse Laplacian operations.
"""

from typing import Any, TypedDict

import jax.numpy as jnp
from jax import Array

from pybasin.solvers.jax_ode_system import JaxODESystem


class RosslerNetworkParams(TypedDict):
    """Parameters for the coupled Rössler network ODE system."""

    a: float  # Rössler parameter a
    b: float  # Rössler parameter b
    c: float  # Rössler parameter c
    K: float  # Coupling strength
    edges_i: Array  # Source node indices for sparse Laplacian
    edges_j: Array  # Target node indices for sparse Laplacian
    N: int  # Number of nodes


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

    Parameters
    ----------
    params : RosslerNetworkParams
        Dictionary containing:
        - a, b, c: Rössler oscillator parameters (typically a=b=0.2, c=7.0)
        - K: Coupling strength (must be in stability interval)
        - edges_i, edges_j: Edge index arrays for sparse Laplacian
        - N: Number of nodes in the network
    """

    def __init__(self, params: RosslerNetworkParams):
        super().__init__(params)
        self._N = params["N"]
        self._edges_i = params["edges_i"]
        self._edges_j = params["edges_j"]

    def ode(self, t: Array, y: Array, args: Any = None) -> Array:
        """
        Right-hand side (RHS) for the coupled Rössler network.

        Parameters
        ----------
        t : Array
            Current time (scalar).
        y : Array
            Current state with shape (3*N,).
            Layout: [x_0..x_{N-1}, y_0..y_{N-1}, z_0..z_{N-1}]

        Returns
        -------
        Array
            Time derivatives with shape (3*N,).
        """
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        k = self.params["K"]
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
