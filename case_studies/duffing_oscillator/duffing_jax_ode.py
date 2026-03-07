"""JAX-native Duffing oscillator ODE system for maximum performance."""

from typing import Any, TypedDict

import jax.numpy as jnp
from jax import Array

from pybasin.solvers.jax_ode_system import JaxODESystem


class DuffingParams(TypedDict):
    """Parameters for the Duffing oscillator ODE system."""

    delta: float  # Damping coefficient
    k3: float  # Cubic stiffness
    A: float  # Forcing amplitude


class DuffingJaxODE(JaxODESystem[DuffingParams]):
    """
    JAX-native Duffing oscillator ODE system for high-performance integration.

    Following Thomson & Steward: Nonlinear Dynamics and Chaos. Page 9, Fig. 1.9.

    The Duffing oscillator dynamics are:
        dx/dt = x_dot
        dx_dot/dt = -delta·x_dot - k3·x³ + A·cos(t)

    Where:
        - x is the displacement
        - x_dot is the velocity
        - delta is the damping coefficient
        - k3 is the cubic stiffness
        - A is the forcing amplitude

    For 5 multistability, recommended parameters are:
        delta = 0.08, k3 = 1, A = 0.2
    """

    def __init__(self, params: DuffingParams):
        super().__init__(params)

    def ode(self, t: Array, y: Array, args: Any = None) -> Array:
        """
        Right-hand side (RHS) for the Duffing oscillator ODE using pure JAX.

        Parameters
        ----------
        t : Array
            Current time (scalar).
        y : Array
            Current state [x, x_dot] with shape (2,).

        Returns
        -------
        Array
            Time derivatives [dx_dt, dx_dot_dt] with shape (2,).
        """
        delta = self.params["delta"]
        k3 = self.params["k3"]
        amplitude = self.params["A"]

        x = y[0]
        x_dot = y[1]

        dx_dt = x_dot
        dx_dot_dt = -delta * x_dot - k3 * x**3 + amplitude * jnp.cos(t)

        return jnp.array([dx_dt, dx_dot_dt])  # pyright: ignore[reportUnknownMemberType]
