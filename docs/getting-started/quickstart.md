# Quick Start

This guide walks through a minimal basin stability estimation using pybasin.

## What is Basin Stability?

Basin stability measures the probability that a dynamical system, starting from a random initial condition, converges to a specific attractor. A system with two attractors (say a fixed point and a limit cycle) might send 60% of random starts toward the fixed point and 40% toward the cycle -- those fractions are the basin stability values.

## Minimal Example

Three objects are required to run an estimation: an ODE system, a sampler, and the estimator itself. Everything else -- solver, feature extractor, clusterer -- has sensible defaults.

### Step 1: Define the ODE system

Subclass `ODESystem` and implement the `ode` method. The Duffing oscillator below is a classic bistable system:

```python
import torch
from typing import TypedDict
from pybasin.ode_system import ODESystem


class DuffingParams(TypedDict):
    delta: float
    k3: float
    A: float


class DuffingODE(ODESystem[DuffingParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = y[..., 0]
        x_dot = y[..., 1]
        delta = self.params["delta"]
        k3 = self.params["k3"]
        A = self.params["A"]
        dx = x_dot
        ddx = -delta * x_dot - k3 * x**3 + A * torch.cos(t)
        return torch.stack([dx, ddx], dim=-1)


params: DuffingParams = {"delta": 0.08, "k3": 1.0, "A": 0.2}
ode_system = DuffingODE(params)
```

### Step 2: Create a sampler

The sampler generates random initial conditions within specified bounds:

```python
from pybasin.sampler import UniformRandomSampler

sampler = UniformRandomSampler(
    min_limits=[-1.0, -0.5],
    max_limits=[1.0, 1.0],
)
```

### Step 3: Estimate basin stability

With those two objects, `BasinStabilityEstimator` handles integration, feature extraction, and clustering automatically:

```python
from pybasin import BasinStabilityEstimator

bse = BasinStabilityEstimator(
    ode_system=ode_system,
    sampler=sampler,
    n=5000,
)

basin_stability = bse.estimate_bs()
print(basin_stability)
# {'attractor_0': 0.52, 'attractor_1': 0.48}
```

The returned dict maps attractor labels to their estimated basin stability fractions. Under the hood, pybasin integrated 5000 trajectories with `TorchDiffEqSolver`, extracted time-series features, and clustered the results with `HDBSCANClusterer`.

## Choosing a Solver

The solver is selected automatically based on the ODE system class:

- **`ODESystem`** (PyTorch) &rarr; `TorchDiffEqSolver` (default, no extras needed)
- **`JaxODESystem`** (JAX) &rarr; `JaxSolver` (requires `pip install pybasin[jax]`)

The Quick Start example above uses `ODESystem`, so `TorchDiffEqSolver` is selected automatically. To use `JaxSolver` for faster GPU integration, define the ODE with `JaxODESystem` instead:

```python
import jax.numpy as jnp
from jax import Array
from pybasin.jax_ode_system import JaxODESystem


class DuffingJaxODE(JaxODESystem[DuffingParams]):
    def ode(self, t: Array, y: Array) -> Array:
        x, x_dot = y[..., 0], y[..., 1]
        dx = x_dot
        ddx = -self.params["delta"] * x_dot - self.params["k3"] * x**3 + self.params["A"] * jnp.cos(t)
        return jnp.stack([dx, ddx], axis=-1)


ode_jax = DuffingJaxODE(params={"delta": 0.08, "k3": 1.0, "A": 0.2})
# JaxSolver is now auto-selected:
bse = BasinStabilityEstimator(ode_system=ode_jax, sampler=sampler, n=5000)
```

You can also pass a solver explicitly to override auto-detection:

```python
from pybasin.solvers import JaxSolver

solver = JaxSolver(time_span=(0, 1000), n_steps=5000, device="cuda")
bse = BasinStabilityEstimator(
    ode_system=ode_jax,
    sampler=sampler,
    solver=solver,
)
```

See the [Solvers guide](../user-guide/solvers.md) for a full comparison of available backends.

## Saving Results

Pass `output_dir` to automatically save results (JSON, Excel, plots) to disk:

```python
bse = BasinStabilityEstimator(
    ode_system=ode_system,
    sampler=sampler,
    output_dir="results/duffing",
)
basin_stability = bse.estimate_bs()
```

## Next Steps

- [Solvers](../user-guide/solvers.md) -- choose the right ODE backend
- [Feature Extractors](../user-guide/feature-extractors.md) -- customize trajectory characterization
- [Case Studies](../case-studies/overview.md) -- validated examples (Duffing, Lorenz, pendulum, friction)
- [API Reference](../api/basin-stability-estimator.md) -- full parameter documentation
