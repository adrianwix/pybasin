from abc import ABC, abstractmethod

import numpy as np


class NumpyODESystem[P](ABC):
    """
    Abstract base class for numpy-based ODE systems, compatible with
    :func:`scipy.integrate.solve_ivp`.

    ``P`` is a type parameter representing the parameter dictionary type.
    Instances are callable and can be passed directly as the ``fun`` argument
    to ``solve_ivp``.
    """

    def __init__(self, params: P) -> None:
        self.params = params

    @abstractmethod
    def ode(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side (RHS) for the ODE.

        :param t: Current time.
        :param y: Current state vector, shape ``(n,)``.
        :return: Time derivatives, shape ``(n,)``.
        """

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.ode(t, y)

    def to(self, *args: object, **kwargs: object) -> "NumpyODESystem[P]":
        """No-op device transfer for compatibility with the ``Solver`` base class."""
        return self
