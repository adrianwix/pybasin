"""JAX-native Lorenz system ODE for maximum performance."""

from typing import TypedDict

import jax.numpy as jnp
from jax import Array

from pybasin.solvers.jax_ode_system import JaxODESystem


class LorenzParams(TypedDict):
    """Parameters for the Lorenz ODE system."""

    sigma: float  # Prandtl number
    r: float  # Rayleigh number
    b: float  # Physical dimension parameter


class LorenzJaxODE(JaxODESystem[LorenzParams]):
    """
    JAX-native Lorenz system ODE for high-performance integration.

    The Lorenz system dynamics are:
        dx/dt = sigma * (y - x)
        dy/dt = r * x - x * z - y
        dz/dt = x * y - b * z

    Classical parameter choice:
        sigma = 10, r = 28, b = 8/3

    For broken butterfly (https://doi.org/10.1142/S0218127414501314):
        sigma = 0.12, r = 0, b = -0.6

    Note: For unbounded trajectory handling, use the event_fn parameter in JaxSolver
    to terminate integration when states exceed a threshold.
    """

    def __init__(self, params: LorenzParams):
        super().__init__(params)

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        """
        Right-hand side of the Lorenz system.

        :param t: Current time, scalar.
        :param y: State vector of shape ``(3,)``, with ``y[0] = x``, ``y[1] = y``, ``y[2] = z``.
        :param p: Parameter array of shape ``(3,)`` ordered as ``[sigma, r, b]``.
        :return: Time derivatives ``[dx_dt, dy_dt, dz_dt]`` of shape ``(3,)``.
        """
        sigma, r, b = p[0], p[1], p[2]

        x = y[0]
        y_ = y[1]  # avoid shadowing y parameter
        z = y[2]

        # Compute standard Lorenz dynamics
        dx_dt = sigma * (y_ - x)
        dy_dt = r * x - x * z - y_
        dz_dt = x * y_ - b * z

        return jnp.array([dx_dt, dy_dt, dz_dt])  # pyright: ignore[reportUnknownMemberType]

    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.
        """
        description = (
            "Lorenz:\n  dx/dt = sigma·(y - x)\n  dy/dt = r·x - x·z - y\n  dz/dt = x·y - b·z\n"
        )
        return description
