# Handling Unbounded Trajectories

## Overview

When computing basin stability, some initial conditions may lead to **unbounded trajectories** that diverge to infinity. These trajectories need special handling to:

1. **Stop integration early** to save computation time
2. **Classify correctly** as a distinct attractor state
3. **Avoid numerical overflow** in the solver

This guide explains the recommended approaches for handling unbounded trajectories in pybasin.

!!! tip "Quick Recommendation"
Use **JaxSolver with event functions** for the best performance and flexibility when dealing with unbounded trajectories.

## Understanding the Problem

In dynamical systems like the Lorenz system, some regions of the state space lead to trajectories that grow without bound:

```python
# Example: Lorenz "broken butterfly" system
params = {"sigma": 0.12, "r": 0.0, "b": -0.6}

# Some initial conditions lead to bounded attractors:
ic_butterfly1 = [0.8, -3.0, 0.0]   # → bounded attractor
ic_butterfly2 = [-0.8, 3.0, 0.0]   # → bounded attractor

# Others lead to unbounded trajectories:
ic_unbounded = [10.0, 50.0, 0.0]   # → |y| → ∞
```

Without proper handling, unbounded trajectories will:

- Waste computation time integrating to large values
- Risk numerical overflow errors
- Contaminate basin stability estimates

---

## Recommended Approach: JaxSolver with Event Functions

The **recommended solution** is to use `JaxSolver` with an event function that stops integration when trajectories exceed a threshold.

### Why JaxSolver?

JAX's `diffrax` library supports **event-based termination** where each trajectory in a batch can stop independently:

- ✅ **Individual termination**: Each trajectory stops when _it_ meets the condition
- ✅ **Efficient**: Other trajectories continue integrating
- ✅ **Clean classification**: Stopped trajectories are marked appropriately

### Complete Example

Here's a complete example from the Lorenz case study:

```python
import torch
from case_studies.lorenz.lorenz_jax_ode import LorenzJaxODE, LorenzParams
from case_studies.lorenz.setup_lorenz_system import lorenz_stop_event
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers.jax_solver import JaxSolver

def main():
    # Number of initial conditions to sample
    n = 10_000

    # Auto-detect device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting up Lorenz system on device: {device}")

    # Parameters for broken butterfly system
    params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}
    ode_system = LorenzJaxODE(params)

    # Sample uniformly in the region of interest
    sampler = UniformRandomSampler(
        min_limits=[-10.0, -20.0, 0.0],
        max_limits=[10.0, 20.0, 0.0],
        device=device
    )

    # JaxSolver with event function to stop unbounded trajectories
    solver = JaxSolver(
        device=device,
        event_fn=lorenz_stop_event,  # ← Key: event function
    )

    # Estimate basin stability
    bse = BasinStabilityEstimator(
        n=n,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        output_dir="results_lorenz",
    )

    basin_stability = bse.estimate_bs()
    print("Basin Stability:", basin_stability)

    return bse

if __name__ == "__main__":
    bse = main()
```

### Defining Event Functions

The event function determines when to stop integration. Here's the `lorenz_stop_event` example:

```python
import jax.numpy as jnp
from diffrax import Event

def lorenz_stop_event(t, y, args):
    """
    Stop integration when trajectory magnitude exceeds 200.

    Args:
        t: Current time
        y: Current state [x, y, z]
        args: Additional arguments (unused)

    Returns:
        Scalar value that triggers event when it crosses zero.
        Negative = continue, positive = stop.
    """
    # Compute maximum absolute value across all state components
    max_magnitude = jnp.max(jnp.abs(y))

    # Return (threshold - magnitude)
    # When magnitude > 200, this becomes negative → event triggers
    return 200.0 - max_magnitude
```

!!! note "Event Function Behavior"
The event triggers when the returned value **crosses zero from positive to negative**. Design your function accordingly:

    - `threshold - magnitude`: triggers when magnitude exceeds threshold
    - `magnitude - threshold`: triggers when magnitude falls below threshold

### Benefits

✅ **Performance**: Only unbounded trajectories stop early  
✅ **Accuracy**: Bounded trajectories integrate to full completion  
✅ **Simplicity**: Clean, declarative API  
✅ **Flexibility**: Custom event functions for any stopping condition

---

## Alternative Approach: Zero Masking with TorchDiffEq

An alternative approach uses **zero masking** where derivatives are set to zero once a stopping condition is met. This "freezes" the solution at the threshold value.

!!! warning "Performance Considerations"
This approach is **slower** than JaxSolver with events because:

    - All trajectories integrate for the full time span (no early stopping)
    - The solver continues stepping even though derivatives are zero
    - Better suited for systems where most trajectories are bounded

### How Zero Masking Works

The ODE system returns zero derivatives when a trajectory exceeds the threshold:

```python
import torch
from pybasin.ode_system import ODESystem

class LorenzODE(ODESystem):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Lorenz dynamics with zero masking for unbounded trajectories.
        """
        # Compute standard Lorenz dynamics
        sigma = self.params["sigma"]
        r = self.params["r"]
        b = self.params["b"]

        x, y_coord, z = y[..., 0], y[..., 1], y[..., 2]

        dx_dt = sigma * (y_coord - x)
        dy_dt = r * x - x * z - y_coord
        dz_dt = x * y_coord - b * z

        dydt = torch.stack([dx_dt, dy_dt, dz_dt], dim=-1)

        # Create mask: 1 if |y| < 200, otherwise 0
        mask = (torch.max(torch.abs(y), dim=-1)[0] < 200).float().unsqueeze(-1)

        # Return masked derivatives (zeros for unbounded trajectories)
        return dydt * mask
```

### Key Points

- **Freezing behavior**: When `|y| ≥ 200`, derivatives become zero, "freezing" the solution
- **No early termination**: Integration continues for the full time span
- **Post-processing**: Feature extraction must detect frozen trajectories by checking final magnitude

### Using with TorchDiffEqSolver

```python
from pybasin.solvers.torchdiffeq_solver import TorchDiffEqSolver

solver = TorchDiffEqSolver(
    device=device,
    t_span=(0.0, 1000.0),
    method="dopri5",
    rtol=1e-8,
    atol=1e-6,
)

# Use with LorenzODE that has zero masking
bse = BasinStabilityEstimator(
    n=n,
    ode_system=LorenzODE(params),  # Has zero masking in ode()
    sampler=sampler,
    solver=solver,
)
```

### Limitations of TorchDiffEq Event Handling

!!! warning "TorchDiffEq Limitation"
`torchdiffeq.odeint_event()` **does not support individual trajectory termination**. When the event condition is met for _any_ trajectory in a batch, **all integrations stop simultaneously**.

    This makes `odeint_event` unsuitable for basin stability estimation where different trajectories should stop at different times.

---

## Comparison

| Feature                    | JaxSolver + Event          | TorchDiffEq + Zero Mask                 |
| -------------------------- | -------------------------- | --------------------------------------- |
| **Individual termination** | ✅ Yes                     | ❌ No (zero masking workaround)         |
| **Performance**            | ✅ Fast (early stopping)   | ⚠️ Slower (full integration)            |
| **Setup complexity**       | 🟢 Simple (event function) | 🟢 Simple (mask in ODE)                 |
| **GPU support**            | ✅ Yes                     | ✅ Yes                                  |
| **Batch processing**       | ✅ Efficient               | ⚠️ Less efficient                       |
| **Best for**               | Most use cases             | Systems with few unbounded trajectories |

---

## Best Practices

### 1. Choose the Right Threshold

Set your stopping threshold based on your system's dynamics:

```python
# Too low: May incorrectly classify bounded trajectories
threshold = 10.0  # ❌ May catch transient behavior

# Good: Well above bounded attractor magnitudes
threshold = 200.0  # ✅ Clear separation

# Check your attractors first:
# - Bounded attractors: |y| < 50
# - Set threshold at 4× max bounded value
```

### 2. Verify Event Triggering

Test your event function with known unbounded initial conditions:

```python
def test_event_function():
    """Verify event triggers for unbounded IC."""
    unbounded_ic = torch.tensor([10.0, 50.0, 0.0])

    solution = solver.solve(
        ode_system=ode_system,
        initial_conditions=unbounded_ic,
    )

    # Check if integration stopped early
    assert solution.t[-1] < t_final, "Event should trigger before t_final"
    assert torch.max(torch.abs(solution.y[-1])) >= threshold
```

### 3. Handle Classification Correctly

Ensure your feature extractor identifies unbounded trajectories:

```python
def extract_features(solution):
    """Extract features, handling unbounded trajectories."""
    max_magnitude = torch.max(torch.abs(solution.y), dim=0)[0]

    if max_magnitude >= 200.0:
        # Unbounded trajectory
        return torch.tensor([0.0, 0.0])  # Special marker
    else:
        # Bounded trajectory - extract features from attractor
        tail = solution.y[-100:]  # Last 100 points
        mean_x = torch.mean(tail[:, 0])

        if mean_x > 0:
            return torch.tensor([1.0, 0.0])  # Attractor 1
        else:
            return torch.tensor([0.0, 1.0])  # Attractor 2
```

---

## Summary

- **Recommended**: Use `JaxSolver` with event functions for efficient, individual trajectory termination
- **Alternative**: Use zero masking with `TorchDiffEqSolver` if you need PyTorch-only solution
- **Avoid**: Using `odeint_event()` for basin stability (stops all trajectories simultaneously)
- **Test**: Always verify your event function or masking logic with known unbounded cases

For more examples, see the [Lorenz case study](../case-studies/lorenz.md).
