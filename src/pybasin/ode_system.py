import ast
import inspect
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any, TypeVar, cast

import torch
import torch.nn as nn

# TypeVar for parameter dictionaries
# Using bound=dict to allow both dict and TypedDict instances
P = TypeVar("P")


class ODESystem[P](ABC, nn.Module):
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

        def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # self.params is typed as MyParams
            alpha = self.params["alpha"]  # type checker knows this exists
            return torch.zeros_like(y)
    ```
    """

    def __init__(self, params: P) -> None:
        # Initialize nn.Module
        super().__init__()  # type: ignore
        self.params = params

    @abstractmethod
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side (RHS) for the ODE using PyTorch tensors.

        :param t: The current time (can be scalar or batch).
        :param y: The current state (can be shape (..., n)).
        :return: The time derivatives with the same leading shape as y.
        """
        pass

    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        By default, auto-generates a representation from the ``ode()`` method source code
        (with docstrings stripped). Override this method to provide a custom representation.

        :return: A human-readable description of the ODE system and its parameters.
        """
        return self._auto_get_str()

    def _auto_get_str(self) -> str:
        """
        Auto-generate string representation from ode() method source code.

        :return: The ode() method source with docstrings stripped, or a fallback string.
        """
        try:
            source = inspect.getsource(self.ode)
            source = dedent(source)
            tree = ast.parse(source)
            func_def = tree.body[0]
            if isinstance(func_def, ast.FunctionDef) and ast.get_docstring(func_def):
                func_def.body = func_def.body[1:]
            return f"{self.__class__.__name__}:\n{ast.unparse(tree)}"
        except (OSError, TypeError, SyntaxError):
            if isinstance(self.params, dict):
                params_dict = cast(dict[str, Any], self.params)  # pyright: ignore[reportUnknownMemberType]
                params_str = ", ".join(f"{k}={v}" for k, v in params_dict.items())
            else:
                params_str = ""
            return f"{self.__class__.__name__}({params_str})"

    # TODO: review if forward is needed for a solver
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calls the ODE function in a manner consistent with nn.Module.
        """
        return self.ode(t, y)
