"""JAX-native Duffing oscillator ODE system for maximum performance."""

from typing import TypedDict

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

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        """
        Right-hand side of the Duffing oscillator ODE.

        :param t: Current time, scalar.
        :param y: State vector of shape ``(2,)``, with ``y[0] = x`` (displacement) and ``y[1] = x_dot`` (velocity).
        :param p: Parameter array of shape ``(3,)`` ordered as ``[delta, k3, A]``.
        :return: Time derivatives ``[dx_dt, dx_dot_dt]`` of shape ``(2,)``.
        """
        delta, k3, amplitude = p[0], p[1], p[2]

        x, x_dot = y[0], y[1]

        dx_dt = x_dot
        dx_dot_dt = -delta * x_dot - k3 * x**3 + amplitude * jnp.cos(t)

        return jnp.array([dx_dt, dx_dot_dt])  # pyright: ignore[reportUnknownMemberType]
