"""JAX-native pendulum ODE system for maximum performance."""

from typing import TypedDict

import jax.numpy as jnp
from jax import Array

from pybasin.solvers.jax_ode_system import JaxODESystem


class PendulumParams(TypedDict):
    """Parameters for the pendulum ODE system."""

    alpha: float  # damping coefficient
    T: float  # external torque
    K: float  # stiffness coefficient


class PendulumJaxODE(JaxODESystem[PendulumParams]):
    """
    JAX-native pendulum ODE system for high-performance integration.

    The pendulum dynamics are:
        dθ/dt = θ̇
        dθ̇/dt = -α·θ̇ + T - K·sin(θ)

    Where:
        - θ is the angle
        - θ̇ is the angular velocity
        - alpha is the damping coefficient
        - T is the external torque
        - K is the stiffness coefficient
    """

    def __init__(self, params: PendulumParams):
        super().__init__(params)

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        """
        Right-hand side of the pendulum ODE.

        :param t: Current time, scalar.
        :param y: State vector of shape ``(2,)``, with ``y[0] = theta`` (angle) and ``y[1] = theta_dot`` (angular velocity).
        :param p: Parameter array of shape ``(3,)`` ordered as ``[alpha, T, K]``.
        :return: Time derivatives ``[dtheta_dt, dtheta_dot_dt]`` of shape ``(2,)``.
        """
        alpha, torque, k = p[0], p[1], p[2]

        theta, theta_dot = y[0], y[1]

        dtheta_dt = theta_dot
        dtheta_dot_dt = -alpha * theta_dot + torque - k * jnp.sin(theta)

        return jnp.array([dtheta_dt, dtheta_dot_dt])  # pyright: ignore[reportUnknownMemberType]
