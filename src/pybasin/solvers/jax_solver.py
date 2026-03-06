# pyright: reportUntypedBaseClass=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Native JAX ODE solver for JaxODESystem.

This module provides a high-performance JAX/Diffrax solver for ODE systems
defined using pure JAX operations. This is the fastest solver option when
using JAX-native ODE systems.
"""

import logging
from collections.abc import Callable
from typing import Any, cast, overload

import jax
import jax.numpy as jnp
import torch
from diffrax import (  # type: ignore[import-untyped]
    Dopri5,
    Event,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
)
from jax import Array

from pybasin.cache_manager import CacheManager
from pybasin.constants import DEFAULT_CACHE_DIR, UNSET
from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.solvers.jax_ode_system import JaxODESystem
from pybasin.utils import DisplayNameMixin, resolve_cache_dir

logger = logging.getLogger(__name__)

DEFAULT_MAX_STEPS: int = 16**5


class JaxSolver(SolverProtocol, DisplayNameMixin):
    """
    High-performance ODE solver using JAX and Diffrax for native JAX ODE systems.

    This solver is optimized for JaxODESystem instances and provides the fastest
    integration performance by avoiding any PyTorch callbacks. It uses JIT
    compilation and vmap for efficient batch processing.

    The interface is compatible with other solvers - it accepts PyTorch tensors
    and returns PyTorch tensors, but internally uses JAX for maximum performance.

    See also: [Diffrax documentation](https://docs.kidger.site/diffrax/)

    Citation:

    ```bibtex
    @phdthesis{kidger2021on,
        title={{O}n {N}eural {D}ifferential {E}quations},
        author={Patrick Kidger},
        year={2021},
        school={University of Oxford},
    }
    ```

    Example usage:

    **Overload 1 — generic API for standard ODEs:**

    ```python
    from pybasin.solvers.jax_ode_system import JaxODESystem
    from pybasin.solvers import JaxSolver
    import torch

    class MyODE(JaxODESystem):
        def ode(self, t, y):
            return -y  # Simple decay
        def get_str(self):
            return "decay"

    solver = JaxSolver(t_span=(0, 10), t_steps=100)
    y0 = torch.tensor([[1.0, 2.0]])  # batch=1, dims=2
    t, y = solver.integrate(MyODE({}), y0)
    ```

    **Overload 2 — direct Diffrax control via ``solver_args``:**

    Pass native Diffrax arguments directly to ``diffeqsolve``. This is useful
    for SDEs, CDEs, or any advanced Diffrax configuration.

    .. note::

       When using ``solver_args``, the integration time points are baked into
       ``saveat.ts`` at construction time. The ``t_steps_factor`` parameter of
       :meth:`clone` has no effect in this mode — the actual integration still
       uses the original ``saveat``.

    ```python
    from diffrax import Dopri5, ODETerm, PIDController, SaveAt
    import jax.numpy as jnp

    solver = JaxSolver(
        solver_args={
            "terms": ODETerm(lambda t, y, args: -y),
            "solver": Dopri5(),
            "t0": 0,
            "t1": 10,
            "dt0": 0.1,
            "saveat": SaveAt(ts=jnp.linspace(0, 10, 100)),
            "stepsize_controller": PIDController(rtol=1e-5, atol=1e-5),
        },
    )
    ```
    """

    @overload
    def __init__(
        self,
        t_span: tuple[float, float] = (0, 1000),
        t_steps: int = 1000,
        device: str | None = None,
        method: Any | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        max_steps: int = DEFAULT_MAX_STEPS,
        event_fn: Callable[[Any, Array, Any], Array] | None = None,
        *,
        t_eval: tuple[float, float] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        solver_args: dict[str, Any],
        cache_dir: str | None = DEFAULT_CACHE_DIR,
    ) -> None: ...

    def __init__(
        self,
        t_span: tuple[float, float] = (0, 1000),
        t_steps: int = 1000,
        device: str | None = None,
        method: Any | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        max_steps: int = DEFAULT_MAX_STEPS,
        event_fn: Callable[[Any, Array, Any], Array] | None = None,
        *,
        solver_args: dict[str, Any] | None = None,
        t_eval: tuple[float, float] | None = None,
    ):
        """
        Initialize JaxSolver.

        Can be called in two ways:

        1. **Generic API** with named parameters for standard ODE integration:

           ``JaxSolver(t_span=(0, 10), t_steps=100, rtol=1e-8, ...)``

        2. **Direct Diffrax control** via ``solver_args`` for full access to
           ``diffeqsolve`` kwargs (SDEs, CDEs, custom step-size controllers, etc.):

           ``JaxSolver(solver_args={"terms": ..., "solver": ..., "t0": ..., ...})``

        The two interfaces are mutually exclusive at the type level.

        :param t_span: Tuple (t_start, t_end) defining the integration interval.
        :param t_steps: Number of evaluation points in the save region.
        :param device: Device to use ('cuda', 'gpu', 'cpu', or None for auto-detect).
        :param method: Diffrax solver instance (e.g., Dopri5(), Tsit5()). Defaults to Dopri5() if None.
        :param rtol: Relative tolerance (used by adaptive-step methods only).
        :param atol: Absolute tolerance (used by adaptive-step methods only).
        :param max_steps: Maximum number of integrator steps.
        :param cache_dir: Directory for caching integration results. Relative paths are
            resolved from the project root. ``None`` disables caching.
        :param event_fn: Optional event function for early termination. Should return positive
                         when integration should continue, negative/zero to stop.
                         Signature: (t, y, args) -> scalar Array.
        :param solver_args: Dict of kwargs passed directly to ``diffrax.diffeqsolve()``.
                            When provided, all other Diffrax-specific parameters are ignored.
                            Must NOT include ``y0`` (provided per-trajectory via ``integrate()``).
        :param t_eval: Optional save region ``(save_start, save_end)``. Only time points
                       in this range are stored. Must be contained within ``t_span``. If ``None``,
                       defaults to ``t_span`` (save all points). Ignored in ``solver_args`` mode.
        """
        self.solver_args: dict[str, Any] | None = solver_args
        if (
            t_eval is not None
            and solver_args is None
            and (t_eval[0] < t_span[0] or t_eval[1] > t_span[1])
        ):
            raise ValueError(f"t_eval {t_eval} must be within t_span {t_span}.")
        self.t_span = t_span
        self.t_steps = t_steps
        self.t_eval = t_eval
        self.event_fn = event_fn
        self.method = method if method is not None else Dopri5()
        self.rtol = rtol
        self.atol = atol
        self.max_steps: int = max_steps
        self._set_device(device)

        self._cache_manager: CacheManager | None = None
        self._cache_dir: str | None = None
        if cache_dir is not None:
            self._cache_dir = resolve_cache_dir(cache_dir)
            self._cache_manager = CacheManager(self._cache_dir)

    def _set_device(self, device: str | None) -> None:
        """
        Set the device for tensor operations with auto-detection.

        :param device: Device to use ('cuda', 'gpu', 'cpu', or None for auto-detect).
        """
        # Store original device string for clone()
        self._device_str = device

        self.jax_device: Any = get_jax_device(device)

        # PyTorch device for output tensors
        self.device = torch.device("cuda:0" if self.jax_device.platform == "gpu" else "cpu")

    def clone(
        self,
        *,
        device: str | None = None,
        t_steps_factor: int = 1,
        cache_dir: str | None | object = UNSET,
    ) -> "JaxSolver":
        """
        Create a copy of this solver, optionally overriding device, resolution, or caching.

        :param device: Target device ('cpu', 'cuda', 'gpu'). If None, keeps the current device.
        :param t_steps_factor: Multiply the number of evaluation points by this factor.
            Ignored for ``solver_args`` mode (saveat is baked in at construction time).
        :param cache_dir: Override cache directory. Pass ``None`` to disable caching.
            If not provided, keeps the current setting.
        :return: New JaxSolver instance.
        """
        effective_cache_dir = self._cache_dir if cache_dir is UNSET else cache_dir
        effective_device = device or self._device_str

        if self.solver_args is not None:
            if t_steps_factor > 1:
                logger.warning(
                    "[JaxSolver] t_steps_factor=%d ignored in solver_args mode "
                    "(saveat is baked in at construction time)",
                    t_steps_factor,
                )
            new_solver = JaxSolver(
                solver_args=self.solver_args,
                cache_dir=effective_cache_dir,  # type: ignore[arg-type]
            )
        else:
            new_solver = JaxSolver(
                t_span=self.t_span,
                t_steps=self.t_steps * t_steps_factor,
                device=effective_device,
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
                cache_dir=effective_cache_dir,  # type: ignore[arg-type]
                event_fn=self.event_fn,
                t_eval=self.t_eval,
            )
        new_solver._set_device(effective_device)
        return new_solver

    def _get_cache_config(self) -> dict[str, Any]:
        """Include solver configuration in cache key."""
        if self.solver_args is not None:
            return {"solver_args": {k: repr(v) for k, v in self.solver_args.items()}}
        return {
            "method": self.method.__class__.__name__,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
            "t_span": self.t_span,
        }

    def integrate(
        self, ode_system: ODESystemProtocol, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.

        :param ode_system: An instance of JaxODESystem.
        :param y0: Initial conditions as PyTorch tensor with shape (batch, n_dims).
        :return: Tuple (t_eval, y_values) as PyTorch tensors where y_values has shape (t_steps, batch, n_dims).
        """
        if y0.ndim != 2:
            raise ValueError(
                f"y0 must be 2D with shape (batch, n_dims), got shape {y0.shape}. "
                f"For single initial condition, use y0.unsqueeze(0) or y0.reshape(1, -1)."
            )

        y0_jax = torch_to_jax(y0, self.jax_device)

        if self.solver_args is not None:
            save_ts_jax = None
        else:
            save_start, save_end = self.t_eval if self.t_eval is not None else self.t_span
            save_ts_jax = jnp.linspace(save_start, save_end, self.t_steps)
            save_ts_jax = jax.device_put(save_ts_jax, self.jax_device)

        def compute() -> tuple[torch.Tensor, torch.Tensor]:
            logger.debug("[%s] Integrating...", self.__class__.__name__)
            ode_system_concrete = cast(JaxODESystem[Any], ode_system)
            t_result_jax, y_result_jax = self._integrate_jax(
                ode_system_concrete, y0_jax, save_ts_jax
            )
            logger.debug("[%s] Integration complete", self.__class__.__name__)
            torch_device = str(y0.device)
            return jax_to_torch(t_result_jax, torch_device), jax_to_torch(
                y_result_jax, torch_device
            )

        if self._cache_manager is not None:
            y0_cpu = y0.detach().cpu()
            save_start_c, save_end_c = self.t_eval if self.t_eval is not None else self.t_span
            save_ts_cpu = torch.linspace(
                float(save_start_c), float(save_end_c), self.t_steps, device="cpu"
            )
            return self._cache_manager.cached_call(
                solver_name=self.__class__.__name__,
                ode_system=ode_system,  # type: ignore[arg-type]
                y0=y0_cpu,
                save_ts=save_ts_cpu,
                solver_config=self._get_cache_config(),
                device=self.device,
                compute_fn=compute,
            )

        logger.debug("[%s] Cache disabled - integrating...", self.__class__.__name__)
        return compute()

    def _integrate_jax(
        self, ode_system: JaxODESystem[Any], y0: Array, save_ts: Array | None
    ) -> tuple[Array, Array]:
        """
        Perform the actual integration using JAX/Diffrax.

        :param ode_system: An instance of JaxODESystem.
        :param y0: Initial conditions as JAX array with shape (batch, n_dims).
        :param save_ts: Save-region time points as JAX array, or None when using solver_args.
        :return: (save_ts, y_values) as JAX arrays.
        """
        if self.solver_args is not None:
            return self._integrate_jax_solver_args(y0)
        assert save_ts is not None
        return self._integrate_jax_generic(ode_system, y0, save_ts)

    def _integrate_jax_generic(
        self, ode_system: JaxODESystem[Any], y0: Array, save_ts: Array
    ) -> tuple[Array, Array]:
        """
        Integration using generic API parameters.

        :param ode_system: An instance of JaxODESystem.
        :param y0: Initial conditions as JAX array with shape (batch, n_dims).
        :param save_ts: Save-region time points as JAX array.
        :return: (save_ts, y_values) as JAX arrays.
        """
        ode_fn = ode_system.ode

        def ode_wrapper(t: Any, y: Array, args: Any) -> Array:
            return ode_fn(t, y)  # type: ignore[arg-type]

        term = ODETerm(ode_wrapper)
        t0 = float(self.t_span[0])
        t1 = float(self.t_eval[1] if self.t_eval is not None else self.t_span[1])
        saveat = SaveAt(ts=save_ts)
        stepsize_controller = PIDController(rtol=self.rtol, atol=self.atol)

        event = Event(cond_fn=self.event_fn) if self.event_fn is not None else None

        def solve_single(y0_single: Array) -> Array:
            sol = diffeqsolve(  # type: ignore[misc]
                term,
                self.method,
                t0=t0,
                t1=t1,
                dt0=None,
                y0=y0_single,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=self.max_steps,
                event=event,
            )
            return sol.ys  # type: ignore[return-value]

        solve_batch = jax.vmap(solve_single)

        try:
            y_batch: Array = solve_batch(y0)
            jax.block_until_ready(y_batch)  # type: ignore[no-untyped-call]
        except Exception as e:
            raise RuntimeError(f"JAX/Diffrax integration failed: {e}") from e

        y_batch_transposed: Array = jnp.transpose(y_batch, (1, 0, 2))  # type: ignore[arg-type]
        return save_ts, y_batch_transposed

    def _integrate_jax_solver_args(self, y0: Array) -> tuple[Array, Array]:
        """
        Integration using raw solver_args passed through to ``diffeqsolve``.

        All Diffrax arguments (including ``terms``) must be provided in
        ``solver_args``. No auto-creation of ``ODETerm`` is performed.

        :param y0: Initial conditions as JAX array with shape (batch, n_dims).
        :return: (t_eval, y_values) as JAX arrays.
        """
        assert self.solver_args is not None
        kwargs: dict[str, Any] = dict(self.solver_args)

        def solve_single(y0_single: Array) -> tuple[Array, Array]:
            sol = diffeqsolve(**kwargs, y0=y0_single)  # type: ignore[misc]
            return sol.ts, sol.ys  # type: ignore[return-value]

        solve_batch = jax.vmap(solve_single)

        try:
            t_batch: Array
            y_batch: Array
            t_batch, y_batch = solve_batch(y0)
            jax.block_until_ready(y_batch)  # type: ignore[no-untyped-call]
        except Exception as e:
            raise RuntimeError(f"JAX/Diffrax integration failed: {e}") from e

        # Transpose from (batch, n_steps, n_dims) to (n_steps, batch, n_dims)
        y_batch_transposed: Array = jnp.transpose(y_batch, (1, 0, 2))  # type: ignore[arg-type]
        # All trajectories share the same time points, take from first
        t_eval: Array = t_batch[0]  # type: ignore[index]

        return t_eval, y_batch_transposed
