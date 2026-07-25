from typing import TypedDict

import torch

from pybasin.solvers.torch_ode_system import ODESystem


class DuffingParams(TypedDict):
    delta: float  # Damping coefficient
    k3: float  # Cubic stiffness
    A: float  # Forcing amplitude


class DuffingODE(ODESystem[DuffingParams]):
    """
    Duffing oscillator ODE system.
    Following Thomson & Steward: Nonlinear Dynamics and Chaos. Page 9, Fig. 1.9.

    For 5 multistability, recommended parameters are:
    delta = 0.08, k3 = 1, A = 0.2
    """

    def __init__(self, params: DuffingParams):
        super().__init__(params)

    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side of the Duffing oscillator ODE.

        :param t: Current time, scalar tensor.
        :param y: State tensor of shape ``(..., 2)``, with ``y[..., 0] = x`` (displacement) and ``y[..., 1] = x_dot`` (velocity).
        :param p: Parameter tensor of shape ``(..., 3)`` ordered as ``[delta, k3, A]``.
        :return: Time derivatives of shape ``(..., 2)``.
        """
        delta, k3, amplitude = p[..., 0], p[..., 1], p[..., 2]
        x = y[..., 0]
        x_dot = y[..., 1]

        dx_dt = x_dot
        dx_dot_dt = -delta * x_dot - k3 * x**3 + amplitude * torch.cos(t)

        return torch.stack([dx_dt, dx_dot_dt], dim=1)
