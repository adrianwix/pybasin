# Installation

## Prerequisites

- Python 3.12 or higher
- pip package manager

## Install from PyPI

```bash
pip install pybasin
```

This installs the core library with `TorchDiffEqSolver` as the default ODE solver backend. For GPU-accelerated JAX integration or other optional features, see the extras below.

## Optional Extras

Certain features rely on packages that are not required for basic usage. Install only what you need:

| Extra         | Install command                    | What it adds                                             |
| ------------- | ---------------------------------- | -------------------------------------------------------- |
| `jax`         | `pip install pybasin[jax]`         | `JaxSolver` -- fastest on GPU, supports event functions  |
| `interactive` | `pip install pybasin[interactive]` | `InteractivePlotter` (Dash/Plotly)                       |
| `tsfresh`     | `pip install pybasin[tsfresh]`     | `TsfreshFeatureExtractor`                                |
| `nolds`       | `pip install pybasin[nolds]`       | `NoldsFeatureExtractor` (nonlinear dynamics features)    |
| `torchode`    | `pip install pybasin[torchode]`    | `TorchOdeSolver` (independent per-trajectory step sizes) |
| `cuml`        | `pip install pybasin[cuml]`        | GPU backend (RAPIDS cuDF/cuML) for clusterers/classifiers |
| `all`         | `pip install pybasin[all]`         | Everything above except `cuml` (multi-gigabyte install)  |

When JAX and Diffrax are installed **and** the ODE system inherits from `JaxODESystem`, `BasinStabilityEstimator` automatically selects `JaxSolver`. If the ODE inherits from `ODESystem` (PyTorch), `TorchDiffEqSolver` is always used -- even when JAX is installed. See the [Solvers guide](../user-guide/solvers.md) for details on ODE system pairing and choosing a backend.

!!! note "JAX GPU/CPU variants"
The `jax` extra installs the default JAX build. If you need CUDA support, follow the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) to install the appropriate GPU build for your system.

!!! note "cuML CUDA variant"
The `cuml` extra pins `cudf-cu13`/`cuml-cu13`, matching the CUDA 13 toolkit pybasin's `torch` dependency requires. On a CUDA 12 system, install `cudf-cu12`/`cuml-cu12` manually instead -- see the [RAPIDS install guide](https://docs.rapids.ai/install/).

## Verification

Confirm everything works:

```python
from pybasin import BasinStabilityEstimator

print("pybasin imported successfully")
```

## Contributing / Install from Source

For development setup, see the [Contributing guide](../development/contributing.md).

## Next Steps

- [Quick Start](quickstart.md) -- run your first basin stability estimation
- [API Reference](../api/basin-stability-estimator.md)
- [Case Studies](../case-studies/overview.md)
