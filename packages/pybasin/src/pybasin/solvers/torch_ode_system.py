from abc import ABC, abstractmethod
from typing import Any, TypeVar, cast

import torch
import torch.nn as nn

from pybasin.utils import AutoGetStrMixin

P = TypeVar("P")


class ODESystem[P](AutoGetStrMixin, ABC, nn.Module):
    """
    Abstract base class for defining an ODE system.

    ``P`` is a type parameter representing the parameter dictionary type.
    Pass a ``TypedDict`` subclass for typed ``self.params`` access.

    ```python
    from typing import TypedDict


    class MyParams(TypedDict):
        alpha: float
        beta: float


    class MyODE(ODESystem[MyParams]):
        def __init__(self, params: MyParams):
            super().__init__(params)

        def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            alpha, beta = p[..., 0], p[..., 1]
            return torch.zeros_like(y)
    ```
    """

    def __init__(self, params: P) -> None:
        # Initialize nn.Module
        super().__init__()  # type: ignore
        self.params = params

    @abstractmethod
    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side (RHS) for the ODE using PyTorch tensors.

        :param t: The current time (can be scalar or batch).
        :param y: The current state (can be shape (..., n)).
        :param p: Flat parameter array with shape ``(n_params,)`` built by
            ``params_to_array()``. Access individual parameters
            via ``p[..., i]`` to support batching.
        :return: The time derivatives with the same leading shape as y.
        """
        pass

    def params_to_array(self) -> torch.Tensor:
        """
        Convert ``self.params`` to a flat tensor.

        Values are ordered by the TypedDict field declaration order.

        :return: Flat tensor of shape ``(n_params,)``.
        """
        p_dict = cast(dict[str, Any], self.params)
        return torch.tensor(list(p_dict.values()))

    # TODO: review if forward is needed for a solver
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calls the ODE function in a manner consistent with nn.Module.

        When ``_batched_params`` is set (by the solver for parameter batching),
        it is used instead of ``params_to_array()``.
        """
        p = getattr(self, "_batched_params", None)
        if p is None:
            p = self.params_to_array()
        return self.ode(t, y, p)
