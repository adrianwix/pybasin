"""Zig native ODE solver package.

Public API:
- :class:`ODEDefinition` — abstract base for ODE definitions
- :class:`ZigODE` — ODE defined in a ``.zig`` source file
- :class:`SymPyODE` — define ODEs symbolically in Python, export to C
- :class:`ZigODESolver` — solver interface (auto-compiles ODEs on demand)
- :func:`list_available_odes` — discover available ODE sources
"""

from ._compiler import list_available_odes, rebuild_solver
from .sympy_ode import SymPyODE
from .zig_solver import ODEDefinition, ZigODE, ZigODESolver

__all__: list[str] = [
    "ODEDefinition",
    "ZigODE",
    "SymPyODE",
    "ZigODESolver",
    "list_available_odes",
    "rebuild_solver",
]
