# pybasin

**Basin stability estimation for dynamical systems**

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

pybasin is a Python library for estimating basin stability in dynamical systems. Given an ODE and a sampling region, it integrates thousands of initial conditions, extracts time-series features, and clusters the resulting trajectories to determine what fraction of the phase space leads to each attractor. The library is a modern port of the MATLAB [bSTAB](https://github.com/TUHH-DYN/bSTAB) toolbox, extended with GPU acceleration, automated feature extraction, and parameter studies.

## Features

- **Basin stability estimation** with configurable sample counts and sampling strategies
- **Multiple ODE solver backends** -- `TorchDiffEqSolver` (default), `JaxSolver` (GPU-optimized, optional), `TorchOdeSolver`, `ScipyParallelSolver`
- **Automated feature extraction** from trajectories (PyTorch-based, plus optional tsfresh and nolds)
- **Unsupervised and supervised classification** -- HDBSCAN clustering out of the box, or bring any scikit-learn estimator
- **Parameter studies** -- sweep ODE parameters and track how basin stability changes
- **Visualization** -- Matplotlib plotting included; interactive Dash/Plotly plotter available via `pybasin[interactive]`
- **Disk caching** -- solver results are cached with safetensors for fast reruns
- **Full type annotations** with `py.typed` marker

## Installation

```bash
pip install pybasin
```

This installs the core library with `TorchDiffEqSolver` as the default backend. Optional extras unlock additional features:

| Extra         | Command                            | Adds                                                          |
| ------------- | ---------------------------------- | ------------------------------------------------------------- |
| `jax`         | `pip install pybasin[jax]`         | `JaxSolver` -- fastest on GPU, event functions                |
| `interactive` | `pip install pybasin[interactive]` | Dash/Plotly interactive plotter                               |
| `tsfresh`     | `pip install pybasin[tsfresh]`     | tsfresh feature extractor                                     |
| `nolds`       | `pip install pybasin[nolds]`       | Nonlinear dynamics features (Lyapunov, correlation dimension) |
| `torchode`    | `pip install pybasin[torchode]`    | torchode solver (per-trajectory step sizes)                   |
| `all`         | `pip install pybasin[all]`         | Everything above                                              |

When JAX is installed **and** the ODE inherits from `JaxODESystem`, `BasinStabilityEstimator` automatically picks `JaxSolver`. If the ODE inherits from `ODESystem` (PyTorch), `TorchDiffEqSolver` is always the default.

## Quick Start

Define an ODE, create a sampler, and run the estimator. Three objects are all you need:

```python
import torch
from typing import TypedDict
from pybasin import BasinStabilityEstimator
from pybasin.ode_system import ODESystem
from pybasin.sampler import UniformRandomSampler


class DuffingParams(TypedDict):
    delta: float
    k3: float
    A: float


class DuffingODE(ODESystem[DuffingParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, x_dot = y[..., 0], y[..., 1]
        delta, k3, A = self.params["delta"], self.params["k3"], self.params["A"]
        return torch.stack([x_dot, -delta * x_dot - k3 * x**3 + A * torch.cos(t)], dim=-1)


ode = DuffingODE({"delta": 0.08, "k3": 1.0, "A": 0.2})
sampler = UniformRandomSampler(min_limits=[-1, -0.5], max_limits=[1, 1])

bse = BasinStabilityEstimator(ode_system=ode, sampler=sampler, n=5000)
basin_stability = bse.estimate_bs()
print(basin_stability)
# {'attractor_0': 0.52, 'attractor_1': 0.48}
```

Solver, feature extractor, and clusterer all have defaults -- you only configure what you need to change. For the full tutorial, see the [Quick Start guide](https://adrianwix.github.io/pybasin/getting-started/quickstart/).

## Documentation

Full documentation: **[https://adrianwix.github.io/pybasin/](https://adrianwix.github.io/pybasin/)**

Covers installation extras, solver comparison, feature extraction, parameter studies, and the complete API reference.

## Case Studies

Validated against the original MATLAB bSTAB implementation:

| Case Study         | Description                         |
| ------------------ | ----------------------------------- |
| Duffing Oscillator | Forced oscillator with bistability  |
| Lorenz System      | Classic chaotic attractor           |
| Pendulum           | Forced pendulum with bifurcations   |
| Friction System    | Mechanical system with friction     |
| Rossler Network    | Coupled oscillators synchronization |

Each case study lives under `case_studies/` and can be run with:

```bash
uv run python -m case_studies.pendulum.main_pendulum_case1
```

## Project Structure

```
root/
├── src/pybasin/          # Published library
├── src/zigode/           # Zig-compiled ODE solver (separate package)
├── thesis_utils/         # Thesis-specific plotting (not part of published package)
├── case_studies/         # Research case studies
├── tests/                # Unit and integration tests
├── docs/                 # Documentation source (MkDocs)
├── benchmarks/           # Performance benchmarks
└── scripts/              # Helper scripts
```

## Development

```bash
git clone https://github.com/adrianwix/pybasin.git
cd pyBasinWorkspace
uv sync --all-groups
```

This installs all dependencies (core + dev + docs + case studies + experiments) into a virtual environment managed by [uv](https://docs.astral.sh/uv/). Run the test suite and linters with:

```bash
uv run pytest                # tests
sh scripts/ci.sh             # ruff + pyright
```

For the full contributor workflow, see the [Contributing guide](https://adrianwix.github.io/pybasin/development/contributing/).

## Academic Context

pybasin is the main contribution of the bachelor thesis "Pybasin: A Python Toolbox for Basin Stability of Multi-Stable Dynamical Systems." It ports and extends the MATLAB bSTAB library with GPU acceleration, automated feature extraction, and modern Python packaging.

## Citation

```bibtex
@software{pybasin2025,
  author = {Wix, Adrian},
  title = {Pybasin: A Python Toolbox for Basin Stability of Multi-Stable Dynamical Systems},
  year = {2025},
  url = {https://github.com/adrianwix/pybasin}
}
```

## License

GPL-3.0 -- see [LICENSE](LICENSE) for details.
