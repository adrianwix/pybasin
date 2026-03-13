from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np

from pybasin.utils import AutoGetStrMixin


class NumpyODESystem[P](AutoGetStrMixin, ABC):
    """
    Abstract base class for numpy-based ODE systems, compatible with
    :func:`scipy.integrate.solve_ivp`.

    ``P`` is a type parameter representing the parameter dictionary type.
    Instances are callable and can be passed directly as the ``fun`` argument
    to ``solve_ivp``.

    Subclasses declare parameters via a ``TypedDict`` type parameter.
    """

    def __init__(self, params: P) -> None:
        self.params = params

    @abstractmethod
    def ode(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Right-hand side (RHS) for the ODE.

        :param t: Current time.
        :param y: Current state vector, shape ``(n,)``.
        :param p: Flat parameter array of shape ``(n_params,)`` built by
            ``params_to_array()``. Access individual parameters via ``p[i]``.
        :return: Time derivatives, shape ``(n,)``.
        """

    def params_to_array(self) -> np.ndarray:
        """Convert ``self.params`` to a flat numpy array.

        Values are ordered by the TypedDict field declaration order.

        :return: Flat array of shape ``(n_params,)``.
        """
        p_dict = cast(dict[str, Any], self.params)
        return np.array(list(p_dict.values()), dtype=np.float64)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.ode(t, y, self.params_to_array())

    def to(self, device: Any) -> "NumpyODESystem[P]":
        """No-op device transfer for compatibility with the ``Solver`` base class."""
        return self
