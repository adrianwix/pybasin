from abc import abstractmethod

import torch

from pybasin.solvers.torch_ode_system import ODESystem


class DiscreteMapSystem[P](ODESystem[P]):
    """Abstract base class for discrete maps ``x_{n+1} = step(x_n, p)``.

    A discrete map has no continuous-time right-hand side, so it does not fit
    ``ODESystem.ode()``; :class:`~pybasin.solvers.discrete_map_solver.DiscreteMapSolver`
    calls :meth:`step` directly instead, once per requested time unit.
    """

    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} has no continuous-time right-hand side -- "
            "DiscreteMapSolver drives iteration via step(), never calling ode()."
        )

    @abstractmethod
    def step(self, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        One map iteration.

        :param y: Current state, shape ``(..., n_dims)``.
        :param p: Flat parameter array with shape ``(n_params,)`` built by
            ``params_to_array()``, or a per-batch-element array with a leading
            batch dimension. Access individual parameters via ``p[..., i]`` to
            support batching, exactly as in :meth:`ODESystem.ode`.
        :return: Next state, same shape as ``y``.
        """
        ...
