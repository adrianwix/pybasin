# pybasin

**Basin stability estimation for dynamical systems**

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

pybasin is a Python library for estimating basin stability in dynamical systems. It's a port of the MATLAB bSTAB library with additional features including parameter studies and neural network-based classification.

## Features

- **Basin Stability Estimation**: Calculate the probability that a system ends up in a specific attractor
- **Parameter Studies**: Systematic parameter sweeps to study how basin stability varies
- **Multiple Solvers**: Support for various ODE solvers including neural ODE
- **Visualization Tools**: Built-in plotting utilities for basin stability results
- **Extensible**: Easy to add custom feature extractors and classifiers

## Installation

```bash
pip install pybasin
```

With optional extras:

```bash
pip install "pybasin[jax]"         # JAX solver
pip install "pybasin[interactive]"  # interactive plotter (Dash)
pip install "pybasin[tsfresh]"     # tsfresh features
pip install "pybasin[nolds]"       # nolds features
pip install "pybasin[all]"         # everything
```

For development:

```bash
git clone https://github.com/adrianwix/pybasin.git
cd pybasinWorkspace
uv sync --all-groups
source .venv/bin/activate
```

## Quick Start

```python
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.solvers.torch_ode_system import ODESystem
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import TorchDiffEqSolver
from pybasin.template_integrator import TemplateIntegrator
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor

class PendulumODE(ODESystem):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        alpha, T, K = self.params["alpha"], self.params["T"], self.params["K"]
        theta, theta_dot = y[..., 0], y[..., 1]
        return torch.stack([
            theta_dot,
            -alpha * theta_dot + T - K * torch.sin(theta),
        ], dim=1)

params = {"alpha": 0.1, "T": 0.5, "K": 1.0}

bse = BasinStabilityEstimator(
    n=1000,
    ode_system=PendulumODE(params),
    sampler=UniformRandomSampler(min_limits=[-np.pi, -10.0], max_limits=[np.pi, 10.0]),
    solver=TorchDiffEqSolver(time_span=(0, 500), n_steps=500),
    feature_extractor=TorchFeatureExtractor(time_steady=450.0, features_per_state={1: {"log_delta": None}}),
    predictor=KNeighborsClassifier(n_neighbors=1),
    template_integrator=TemplateIntegrator(
        template_y0=[[0.4, 0.0], [2.7, 0.0]],
        labels=["FP", "LC"],
        ode_params=params,
    ),
)

result = bse.estimate_bs()
print(result["basin_stability"])
```

## Documentation

Full documentation is available at [https://adrianwix.github.io/pybasin/](https://adrianwix.github.io/pybasin/)

## Case Studies

This repository includes several case studies from the original bSTAB paper:

- **Duffing Oscillator**: Forced oscillator with two attractors
- **Lorenz System**: Classic chaotic system
- **Pendulum**: Forced pendulum with bifurcations
- **Friction System**: System with friction effects

See the [Case Studies](https://adrianwix.github.io/pybasin/case-studies/overview/) documentation for details.

## Benchmarks

Performance benchmarks comparing solvers, and end-to-end throughput are available in the [Benchmarks](https://adrianwix.github.io/pybasin/benchmarks/overview/) section.

## Project Structure

```
pybasinWorkspace/
├── src/pybasin/          # Main library (published to PyPI)
├── src/zigode/           # Native Zig ODE solver (separate package)
├── case_studies/         # Research case studies
├── thesis_utils/         # Thesis-specific plotting utilities
├── benchmarks/           # Performance benchmarks
├── tests/                # Unit and integration tests
└── docs/                 # Documentation source
```

## Development

### Setup

```bash
uv sync --all-groups
```

### Running Tests

```bash
uv run pytest
```

### CI (lint, types, tests)

```bash
bash scripts/ci.sh
```

### Building Documentation

```bash
uv run mkdocs serve  # Local preview
uv run mkdocs build  # Build static site
```

## Related Projects

- **bSTAB**: Original MATLAB implementation - [GitHub](https://github.com/original/bSTAB)

## Citation

If you use pybasin in your research, please cite:

```bibtex
@software{pybasin2025,
  author = {Wix, Adrian},
  title = {pybasin: Basin Stability Estimation for Dynamical Systems},
  year = {2025},
  url = {https://github.com/adrianwix/pybasin}
}
```

## License

This project is licensed under the GPL-3.0-or-later License - see the [LICENSE](https://github.com/adrianwix/pybasin/blob/main/LICENSE) file for details.

## Acknowledgments

- Based on the bSTAB MATLAB library
- Part of a bachelor thesis on basin stability estimation
