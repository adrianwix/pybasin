"""ODE solvers for pybasin.

This package provides ODE solvers for numerical integration of dynamical systems.
The default solver is ``TorchDiffEqSolver`` (torchdiffeq). If JAX and Diffrax are
installed (``pip install pybasin[jax]``), ``JaxSolver`` becomes available and is
automatically selected as the default when using ``JaxODESystem``.

Solvers
-------
TorchDiffEqSolver : Default PyTorch-based solver (torchdiffeq)
    Works out of the box. Fastest on CPU at large N.

ScipyParallelSolver : SciPy-based solver with joblib parallelization
    Useful for debugging and reference baselines.

JaxSolver : JAX/Diffrax solver (requires ``pybasin[jax]``)
    Fastest on GPU. Supports event-based early termination.

TorchOdeSolver : torchode solver (requires ``pybasin[torchode]``)
    Independent per-trajectory step sizes.
"""

from pybasin.solvers.scipy_solver import ScipyParallelSolver
from pybasin.solvers.torchdiffeq_solver import TorchDiffEqSolver

__all__: list[str] = ["ScipyParallelSolver", "TorchDiffEqSolver"]

try:
    from pybasin.solvers.jax_solver import JaxSolver

    __all__.append("JaxSolver")
except ImportError:
    pass

try:
    from pybasin.solvers.torchode_solver import TorchOdeSolver

    __all__.append("TorchOdeSolver")
except ImportError:
    pass
