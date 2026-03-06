# Solvers

## ODE System Classes

pybasin provides three ODE system base classes, each paired with specific solver backends:

- **`ODESystem`** -- PyTorch-based. Define `ode(t, y)` using `torch` operations. Works with `TorchDiffEqSolver` and `TorchOdeSolver`.
- **`JaxODESystem`** -- JAX-based. Define `ode(t, y)` using `jax.numpy` operations. Works with `JaxSolver` for JIT-compiled, GPU-optimized integration.
- **`NumpyODESystem`** -- NumPy-based. Define `ode(t, y)` using `numpy` operations. Required by `ScipyParallelSolver`; instances are callable and pass directly to `scipy.integrate.solve_ivp` as the `fun` argument.

When `solver=None`, `BasinStabilityEstimator` auto-selects the solver based on which class the ODE inherits from. See the [Solvers user guide](../user-guide/solvers.md#ode-system-pairing) for details.

::: pybasin.solvers.torch_ode_system.ODESystem

---

::: pybasin.solvers.numpy_ode_system.NumpyODESystem

---

::: pybasin.solvers.jax_ode_system.JaxODESystem

---

## Solver Protocol

::: pybasin.protocols.SolverProtocol

---

## Solver Implementations

::: pybasin.solvers.jax_solver.JaxSolver

---

::: pybasin.solvers.TorchDiffEqSolver

---

::: pybasin.solvers.TorchOdeSolver

---

::: pybasin.solvers.ScipyParallelSolver
