from typing import TypedDict

import numpy as np
import torch

from pybasin.solvers.numpy_ode_system import NumpyODESystem
from pybasin.solvers.torch_ode_system import ODESystem


class PendulumParams(TypedDict):
    """Parameters for the pendulum ODE system."""

    alpha: float  # damping coefficient
    T: float  # external torque
    K: float  # stiffness coefficient


class PendulumODE(ODESystem[PendulumParams]):
    def __init__(self, params: PendulumParams):
        super().__init__(params)

    # TODO: Remove t from the signature if not used
    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side of the pendulum ODE.

        :param t: Current time, scalar tensor.
        :param y: State tensor of shape ``(..., 2)``, with ``y[..., 0] = theta`` (angle) and ``y[..., 1] = theta_dot`` (angular velocity).
        :param p: Parameter tensor of shape ``(..., 3)`` ordered as ``[alpha, T, K]``.
        :return: Time derivatives of shape ``(..., 2)``.
        """
        alpha, torque, k = p[..., 0], p[..., 1], p[..., 2]

        theta = y[..., 0]
        theta_dot = y[..., 1]

        dtheta_dt = theta_dot
        dtheta_dot_dt = -alpha * theta_dot + torque - k * torch.sin(theta)

        return torch.stack([dtheta_dt, dtheta_dot_dt], dim=1)


class PendulumNumpyODE(NumpyODESystem[PendulumParams]):
    def ode(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        alpha, torque, k = p[0], p[1], p[2]
        return np.array([y[1], -alpha * y[1] + torque - k * np.sin(y[0])])
