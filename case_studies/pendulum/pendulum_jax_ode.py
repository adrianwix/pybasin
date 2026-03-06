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

    def ode(self, t: Array, y: Array) -> Array:
        """
        Right-hand side (RHS) for the pendulum ODE using pure JAX.

        Parameters
        ----------
        t : Array
            Current time (scalar).
        y : Array
            Current state [theta, theta_dot] with shape (2,).

        Returns
        -------
        Array
            Time derivatives [dtheta_dt, dtheta_dot_dt] with shape (2,).
        """
        alpha = self.params["alpha"]
        torque = self.params["T"]
        k = self.params["K"]

        theta = y[0]
        theta_dot = y[1]

        dtheta_dt = theta_dot
        dtheta_dot_dt = -alpha * theta_dot + torque - k * jnp.sin(theta)

        return jnp.array([dtheta_dt, dtheta_dot_dt])  # pyright: ignore[reportUnknownMemberType]
