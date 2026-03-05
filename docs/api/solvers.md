# Solvers

## ODE System Classes

pybasin provides two ODE system base classes, each paired with specific solver backends:

- **`ODESystem`** -- PyTorch-based. Define `ode(t, y)` using `torch` operations. Works with `TorchDiffEqSolver`, `TorchOdeSolver`, and `ScipyParallelSolver`.
- **`JaxODESystem`** -- JAX-based. Define `ode(t, y)` using `jax.numpy` operations. Works with `JaxSolver` for JIT-compiled, GPU-optimized integration.

When `solver=None`, `BasinStabilityEstimator` auto-selects the solver based on which class the ODE inherits from. See the [Solvers user guide](../user-guide/solvers.md#ode-system-pairing) for details.

::: pybasin.ode_system.ODESystem

---

::: pybasin.jax_ode_system.JaxODESystem

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
