# pyright: reportUntypedBaseClass=false
"""JAX-native ODE system base class.

This module provides a base class for defining ODE systems using pure JAX operations,
enabling JIT compilation and efficient GPU execution without PyTorch callbacks.
"""

from typing import Any, TypeVar, cast

import jax.numpy as jnp
from jax import Array

from pybasin.utils import AutoGetStrMixin

# TypeVar for parameter dictionaries
P = TypeVar("P")


class JaxODESystem[P](AutoGetStrMixin):
    """
    Base class for defining an ODE system using pure JAX.

    This class is designed for ODE systems that need maximum performance with JAX/Diffrax.
    Unlike the PyTorch-based ODESystem, this uses pure JAX operations that can be
    JIT-compiled for optimal GPU performance.

    ``P`` is a type parameter representing the parameter dictionary type.
    Pass a ``TypedDict`` subclass for typed ``self.params`` access.

    For standard ODEs, subclass and override ``ode()``:

    ```python
    from typing import TypedDict
    import jax.numpy as jnp
    from jax import Array


    class MyParams(TypedDict):
        alpha: float
        beta: float


    class MyJaxODE(JaxODESystem[MyParams]):
        def __init__(self, params: MyParams):
            super().__init__(params)

        def ode(self, t: Array, y: Array, p: Array) -> Array:
            alpha, beta = p[0], p[1]
            return jnp.zeros_like(y)
    ```

    For SDEs or CDEs where you provide custom Diffrax ``terms`` via ``solver_args``,
    overriding ``ode()`` is not required. The subclass only needs ``params`` and
    ``get_str()`` for caching and display:

    ```python
    class MySDESystem(JaxODESystem[MyParams]):
        def __init__(self, params: MyParams):
            super().__init__(params)

        def get_str(self) -> str:
            return f"MySDE(alpha={self.params['alpha']})"
    ```
    """

    def __init__(self, params: P) -> None:
        """
        Initialize the JAX ODE system.

        :param params: Dictionary of ODE parameters.
        """
        self.params = params

    def ode(self, t: Array, y: Array, p: Array) -> Array:
        """
        Right-hand side (RHS) for the ODE using pure JAX operations.

        Override this method for standard ODE systems. For SDEs or CDEs where
        custom Diffrax terms are provided via ``JaxSolver(solver_args=...)``,
        overriding this method is not required.

        This method must use only JAX operations (jnp, not np or torch)
        to enable JIT compilation and efficient execution.

        Notes:

        - Use jnp operations instead of np or torch
        - Avoid Python control flow that depends on array values
        - This method will be JIT-compiled, so ensure it's traceable
        - ``p`` is the flat parameter array built from ``params_to_array()``. Access
          individual parameters via ``p[i]``. Batching is handled by ``vmap``,
          so the ODE always sees unbatched 1-D arrays.

        :param t: The current time (scalar JAX array).
        :param y: The current state with shape (n_dims,) for single trajectory.
        :param p: Flat parameter array with shape (n_params,), built by
            ``params_to_array()``.
        :return: The time derivatives with the same shape as y.
        :raises NotImplementedError: If not overridden and called directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.ode() is not implemented. "
            "Override ode() for standard ODEs, or provide custom Diffrax terms "
            "via JaxSolver(solver_args={'terms': ...})."
        )

    def params_to_array(self) -> Array:
        """
        Convert ``self.params`` to a flat JAX array.

        Values are ordered by the TypedDict field declaration order.

        :return: Flat JAX array of shape ``(n_params,)``.
        """
        p_dict = cast(dict[str, Any], self.params)
        return jnp.array(list(p_dict.values()))  # pyright: ignore[reportUnknownMemberType]

    def to(self, device: Any) -> "JaxODESystem[P]":
        """No-op for JAX systems - device handling is done on tensors.

        This method exists for API compatibility with PyTorch-based ODESystem.

        :param device: Ignored for JAX systems.
        :return: Returns self.
        """
        return self

    def __call__(self, t: Array, y: Array, args: Any = None) -> Array:
        """
        Make the ODE system callable for use with Diffrax.

        Diffrax expects f(t, y, args) signature.

        :param t: Current time.
        :param y: Current state.
        :param args: Passed through to ode().
        :return: Time derivatives.
        """
        return self.ode(t, y, args)
