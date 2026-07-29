"""Zig ODE Solver Python Interface.

Provides:

- :class:`ODEDefinition` — abstract base for ODE definitions
- :class:`ZigODE` — ODE defined in a ``.zig`` source file
- :class:`ZigODESolver` — solver that compiles and runs any :class:`ODEDefinition`

Usage::

    from zigode import ZigODE, ZigODESolver

    pendulum = ZigODE(name="pendulum", param_names=["alpha", "T", "K"])
    solver = ZigODESolver()
    t, y = solver.solve(
        pendulum,
        y0=[0.5, 0.0],
        t_span=(0.0, 10.0),
        t_eval=np.linspace(0, 10, 100),
        params={"alpha": 0.1, "T": 0.5, "K": 1.0},
    )
"""

from __future__ import annotations

import abc
import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._compiler import (
    compile_ode_if_needed,
    get_ode_lib,
    ode_needs_compile,
    rebuild_solver,
    solver_needs_rebuild,
)
from ._paths import LIB_DIR, SOLVER_LIB_NAME, USER_ODE_DIR

# ============================================================
# ODE Definition Interface
# ============================================================


class ODEDefinition(abc.ABC):
    """Abstract base class for an ODE that can be solved by :class:`ZigODESolver`.

    Subclasses define how the ODE source is produced (hand-written Zig or
    generated C) and declare parameter names so that a ``params`` dict can
    be mapped to the C ``Params`` struct regardless of key order.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier used as the filename stem and cache key."""

    @property
    @abc.abstractmethod
    def param_names(self) -> list[str]:
        """Parameter names in the order they appear in the C ``Params`` struct."""

    @abc.abstractmethod
    def source_exists(self) -> bool:
        """Return True if the ODE source file is present on disk."""

    @abc.abstractmethod
    def ensure_source(self) -> Path:
        """Ensure the source file exists, creating or updating it if needed.

        :return: Path to the source file.
        :raises FileNotFoundError: If the source cannot be produced.
        """

    def build_params_ptr(self, params: dict[str, float]) -> ctypes.c_void_p:
        """Build a ctypes pointer from *params*, mapped by name.

        Keys are matched against :attr:`param_names` so the caller does not
        need to care about dict insertion order.

        :param params: Parameter values keyed by name.
        :return: ``ctypes.c_void_p`` pointing to the packed C struct.
        :raises ValueError: If keys don't match :attr:`param_names` exactly.
        """
        expected = set(self.param_names)
        provided = set(params.keys())
        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            parts: list[str] = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra:
                parts.append(f"unexpected: {extra}")
            raise ValueError(f"Params mismatch for '{self.name}': {', '.join(parts)}")

        fields: list[tuple[str, type]] = [(n, ctypes.c_double) for n in self.param_names]
        param_type = type("Params", (ctypes.Structure,), {"_fields_": fields})
        ordered: dict[str, float] = {n: params[n] for n in self.param_names}
        struct = param_type(**ordered)
        return ctypes.cast(ctypes.pointer(struct), ctypes.c_void_p)


class ZigODE(ODEDefinition):
    """An ODE defined in a hand-written ``.zig`` file in ``user_odes/``.

    :param name: Filename stem (e.g. ``"pendulum"`` for ``pendulum.zig``).
    :param param_names: Parameter names in the order they appear in the
        Zig ``Params`` extern struct.
    """

    def __init__(self, name: str, param_names: list[str]) -> None:
        self._name = name
        self._param_names = param_names

    @property
    def name(self) -> str:
        return self._name

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    def source_exists(self) -> bool:
        return (USER_ODE_DIR / f"{self._name}.zig").exists()

    def ensure_source(self) -> Path:
        """Verify the ``.zig`` source exists.

        :return: Path to the source file.
        :raises FileNotFoundError: If the source is missing.
        """
        path = USER_ODE_DIR / f"{self._name}.zig"
        if not path.exists():
            raise FileNotFoundError(f"Zig ODE source not found: {path}")
        return path


# ============================================================
# Loaded ODE (internal)
# ============================================================


class _LoadedODE:
    """A loaded ODE shared library with its function pointer."""

    def __init__(self, name: str, lib: ctypes.CDLL) -> None:
        self.name = name
        self._lib = lib

        self._lib.ode_func.argtypes = [
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.ode_func.restype = None

        self._lib.ode_dim.argtypes = []
        self._lib.ode_dim.restype = ctypes.c_size_t

        self._lib.ode_param_size.argtypes = []
        self._lib.ode_param_size.restype = ctypes.c_size_t

    @property
    def dim(self) -> int:
        return self._lib.ode_dim()

    @property
    def param_size(self) -> int:
        return self._lib.ode_param_size()

    @property
    def func_ptr(self) -> ctypes.c_void_p:
        """Raw function pointer to pass to the solver."""
        return ctypes.cast(self._lib.ode_func, ctypes.c_void_p)


# ============================================================
# Solver
# ============================================================


class ZigODESolver:
    """Python interface to the Zig Dopri5 ODE solver.

    The solver library is loaded once. ODE libraries are compiled and loaded
    on-demand when first used.
    """

    def __init__(self, auto_rebuild_solver: bool = True) -> None:
        """Initialise the solver.

        :param auto_rebuild_solver: Rebuild solver library if sources changed.
        """
        if auto_rebuild_solver and solver_needs_rebuild():
            rebuild_solver()

        lib_path = LIB_DIR / SOLVER_LIB_NAME
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Solver library not found: {lib_path}\nRun rebuild_solver() to build it."
            )

        self._solver_lib = ctypes.CDLL(str(lib_path))
        self._setup_solver_functions()
        self._ode_cache: dict[str, _LoadedODE] = {}

    def _setup_solver_functions(self) -> None:
        """Set up ctypes signatures for solver library."""
        self._solver_lib.solve_ode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._solver_lib.solve_ode.restype = ctypes.c_int32

        self._solver_lib.solve_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self._solver_lib.solve_batch.restype = ctypes.c_int32

    def _load_ode(self, ode: ODEDefinition) -> _LoadedODE:
        """Ensure ODE source exists, compile if needed, and load the shared library."""
        name = ode.name
        ode.ensure_source()

        if name in self._ode_cache and not ode_needs_compile(name):
            return self._ode_cache[name]

        compile_ode_if_needed(name)

        lib_path = get_ode_lib(name)
        if not lib_path.exists():
            raise FileNotFoundError(f"ODE library not found: {lib_path}")

        ode_lib = ctypes.CDLL(str(lib_path))
        loaded = _LoadedODE(name, ode_lib)
        self._ode_cache[name] = loaded
        return loaded

    def solve(
        self,
        ode: ODEDefinition,
        y0: NDArray[np.float64] | list[float] | list[list[float]],
        t_span: tuple[float, float],
        t_eval: NDArray[np.float64] | list[float],
        params: dict[str, float] | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        max_steps: int = 1_000_000,
        n_threads: int = 0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve an ODE system for one or many initial conditions.

        :param ode: The ODE definition to solve.
        :param y0: Initial conditions — ``(dim,)`` or ``(n_ics, dim)``.
        :param t_span: Integration interval ``(t0, t1)``.
        :param t_eval: Times at which to store the solution.
        :param params: ODE parameters (key order does not matter).
        :param rtol: Relative tolerance.
        :param atol: Absolute tolerance.
        :param max_steps: Maximum number of integration steps.
        :param n_threads: Number of threads (0 = auto-detect CPU count).
        :return: ``(t_eval, y)`` where ``y`` has shape ``(n_ics, n_save, dim)``.
        """
        loaded = self._load_ode(ode)
        dim = loaded.dim

        y0_arr = np.ascontiguousarray(y0, dtype=np.float64)
        t_eval_arr = np.asarray(t_eval, dtype=np.float64)

        if y0_arr.ndim == 1:
            if y0_arr.shape[0] != dim:
                raise ValueError(f"y0 has length {y0_arr.shape[0]}, expected {dim}")
            y0_arr = y0_arr.reshape(1, dim)
        elif y0_arr.ndim == 2:
            if y0_arr.shape[1] != dim:
                raise ValueError(f"y0 second axis is {y0_arr.shape[1]}, expected {dim}")
        else:
            raise ValueError(f"y0 must be 1-D or 2-D, got {y0_arr.ndim}-D")

        n_ics = y0_arr.shape[0]
        n_save = len(t_eval_arr)

        params_ptr = ode.build_params_ptr(params) if params is not None else None

        results = np.zeros((n_ics, n_save, dim), dtype=np.float64)

        t0, t1 = t_span
        error_code = self._solver_lib.solve_batch(
            loaded.func_ptr,
            y0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_ics,
            dim,
            t0,
            t1,
            t_eval_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_save,
            rtol,
            atol,
            max_steps,
            params_ptr,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_threads,
        )

        if error_code != 0:
            error_messages: dict[int, str] = {
                -1: "Invalid ODE function pointer",
                -2: "Memory allocation failed",
                -3: "Solver failed",
                -4: "Thread spawn failed",
            }
            msg = error_messages.get(error_code, f"Unknown error {error_code}")
            raise RuntimeError(f"Zig solver error: {msg}")

        return t_eval_arr, results
