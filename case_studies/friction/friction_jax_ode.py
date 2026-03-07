"""JAX-native friction ODE system for maximum performance."""

from typing import Any, TypedDict

import jax.numpy as jnp
from jax import Array

from pybasin.solvers.jax_ode_system import JaxODESystem


class FrictionParams(TypedDict):
    """Parameters for the friction ODE system."""

    v_d: float  # Driving velocity
    xi: float  # Damping ratio
    musd: float  # Ratio of static to dynamic friction coefficient
    mud: float  # Dynamic friction coefficient
    muv: float  # Linear strengthening parameter
    v0: float  # Reference velocity for exponential decay


class FrictionJaxODE(JaxODESystem[FrictionParams]):
    """
    JAX-native friction-based SDOF oscillator ODE system for high-performance integration.

    Adapted from ode_friction.m.

    The friction oscillator dynamics involve:
        - Displacement (disp)
        - Velocity (vel)
        - Stick-slip friction behavior with smooth transitions
    """

    def __init__(self, params: FrictionParams):
        super().__init__(params)

    def ode(self, t: Array, y: Array, args: Any = None) -> Array:
        """
        Right-hand side (RHS) for the friction ODE using pure JAX.

        Parameters
        ----------
        t : Array
            Current time (scalar).
        y : Array
            Current state [disp, vel] with shape (2,).

        Returns
        -------
        Array
            Time derivatives [ddisp_dt, dvel_dt] with shape (2,).
        """
        v_d = self.params["v_d"]
        xi = self.params["xi"]
        musd = self.params["musd"]
        mud = self.params["mud"]
        muv = self.params["muv"]
        v0 = self.params["v0"]

        disp = y[0]
        vel = y[1]

        vrel = vel - v_d
        eta = 1e-4
        k_smooth = (
            200.0  # Controls transition smoothness (higher = sharper, closer to hard switching)
        )

        # Friction force calculation
        f_fric = mud + mud * (musd - 1.0) * jnp.exp(-jnp.abs(vrel) / v0) + muv * jnp.abs(vrel) / v0

        slip_condition = jnp.abs(vrel) > eta
        trans_condition = jnp.abs(disp + 2 * xi * vel) > mud * musd

        smooth_sign_vrel = jnp.tanh(k_smooth * vrel)
        smooth_sign_ffric = jnp.tanh(k_smooth * f_fric)

        slip_vel = -disp - 2 * xi * vel - smooth_sign_vrel * f_fric
        trans_vel = -disp - 2 * xi * vel + mud * musd * smooth_sign_ffric
        stick_vel = -(vel - v_d)

        ddisp_dt = jnp.where(slip_condition | trans_condition, vel, v_d)
        dvel_dt = jnp.where(
            slip_condition, slip_vel, jnp.where(trans_condition, trans_vel, stick_vel)
        )

        return jnp.array([ddisp_dt, dvel_dt])  # pyright: ignore[reportUnknownMemberType]
