# TorchOdeSolver - Alternative ODE Solver

This document explains the `TorchOdeSolver` implementation, an alternative to `TorchDiffEqSolver`.

## Overview

`TorchOdeSolver` is a PyTorch-based ODE solver that uses the [torchode](https://github.com/martenlienen/torchode) library. It provides:

- **JIT Compilation**: Optional PyTorch JIT compilation for better performance
- **Batch Parallelization**: Efficient parallel solving across batches
- **Multiple Methods**: Various integration methods (dopri5, tsit5, euler, etc.)
- **GPU Support**: Full CUDA support like TorchDiffEqSolver

## Performance Comparison

⚠️ **Important**: In the current implementation, **TorchDiffEqSolver is faster** than TorchOdeSolver for single-trajectory integration:

- **TorchDiffEqSolver**: ~76 seconds for 10,000 samples (pendulum case study)
- **TorchOdeSolver**: ~119 seconds for 10,000 samples (pendulum case study)

This is because:

1. The current architecture integrates one trajectory at a time (batch_size=1)
2. torchode's batch parallelization doesn't help with batch_size=1
3. torchdiffeq is more optimized for single-trajectory integration

**When TorchOdeSolver would be faster**:

- When integrating **multiple trajectories in parallel** (requires code restructuring)
- When using **JIT compilation** with repeated solves of the same system
- For problems where **variable step sizes per trajectory** are needed

## Installation

To use `TorchOdeSolver`, you need to install the `torchode` package:

```bash
# Using pip
pip install torchode

# Using uv
uv add torchode

# Or install with the optional solvers dependency
pip install -e ".[solvers]"
```

## Comparison: TorchDiffEqSolver vs TorchOdeSolver

### TorchDiffEqSolver (torchdiffeq)

- **Default Solver**: dopri5 (Dormand-Prince 5(4))
- **Similar to**: MATLAB's ode45
- **Pros**:
  - Well-established, widely used
  - Adjoint method for memory-efficient backpropagation
  - Simple API
- **Cons**:
  - No batch parallelization
  - No JIT compilation support

### TorchOdeSolver (torchode)

- **Default Solver**: dopri5 (Dormand-Prince 5(4))
- **Pros**:
  - JIT compilation support for performance
  - Batch parallelization (different step sizes per sample)
  - Modern PyTorch integration
  - Multiple solver methods
- **Cons**:
  - Newer library, less widespread adoption
  - More complex API

## Available Methods

### Adaptive-Step Methods

- **`dopri5`** (default): Dormand-Prince 5(4) - similar to MATLAB's ode45
- **`tsit5`**: Tsitouras 5(4) - often more efficient than dopri5

### Fixed-Step Methods

- **`euler`**: Explicit Euler (1st order)
- **`midpoint`**: Explicit midpoint (2nd order)
- **`heun`**: Heun's method (2nd order)

## Usage Example

### Basic Usage

```python
from pybasin.solvers import TorchOdeSolver

# Create solver with default settings (dopri5)
solver = TorchOdeSolver(time_span=(0, 1000), n_steps=25001, device="cuda")
```

### With Custom Settings

```python
solver = TorchOdeSolver(
    time_span=(0, 1000),
    n_steps=25001,
    device="cuda",
    method="tsit5",  # Use Tsitouras method
    rtol=1e-8,  # Relative tolerance
    atol=1e-6,  # Absolute tolerance
    use_jit=True,  # Enable JIT compilation
)
```

### Complete Example

See `case_studies/pendulum/main_pendulum_case1_torchode.py` for a complete working example.

```python
from case_studies.pendulum.setup_pendulum_system_torchode import (
    setup_pendulum_system_torchode,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator

# Setup system with TorchOdeSolver
props = setup_pendulum_system_torchode()

# Create estimator
bse = BasinStabilityEstimator(
    n=props["n"],
    ode_system=props["ode_system"],
    sampler=props["sampler"],
    solver=props["solver"],  # Using TorchOdeSolver
    feature_extractor=props["feature_extractor"],
    estimator=props["estimator"],
)

# Estimate basin stability
basin_stability = bse.estimate_bs()
```

## Running the Test Case Study

To test the TorchOdeSolver implementation:

```bash
# First, install torchode
uv add torchode

# Run the pendulum case study with TorchOdeSolver
python case_studies/pendulum/main_pendulum_case1_torchode.py
```

## Performance Tips

1. **Enable JIT Compilation**: Set `use_jit=True` for repeated solves with the same system

   ```python
   solver = TorchOdeSolver(time_span=(0, 1000), n_steps=25001, use_jit=True)
   ```

2. **Choose the Right Method**:
   - For general problems: `dopri5` (default)
   - For better efficiency: `tsit5`
   - For simple/fast problems: `euler` or `midpoint` (fixed step)

3. **Adjust Tolerances**:
   - Tighter tolerances (smaller rtol/atol) = more accurate but slower
   - Looser tolerances = faster but less accurate

4. **GPU Acceleration**: Always specify `device="cuda"` if available

## Implementation Details

The `TorchOdeSolver` class:

- Inherits from the abstract `Solver` base class
- Implements the `_integrate()` method
- Handles batch dimension conversion (torchode expects batched inputs)
- Supports caching like TorchDiffEqSolver
- Falls back gracefully if torchode is not installed

## Troubleshooting

### Import Error

```
ImportError: torchode is not installed
```

**Solution**: Install torchode with `pip install torchode`

### Unknown Method Error

```
ValueError: Unknown method: xyz
```

**Solution**: Use one of the available methods: dopri5, tsit5, euler, midpoint, heun

### Integration Failed

```
RuntimeError: torchode integration failed
```

**Solution**: Try adjusting tolerances or using a different method

## References

- [torchode GitHub](https://github.com/martenlienen/torchode)
- [torchode Documentation](https://torchode.readthedocs.io/)
- [torchode Paper](https://openreview.net/forum?id=uiKVKTiUYB0)
- [torchdiffeq GitHub](https://github.com/rtqichen/torchdiffeq) (for comparison)
