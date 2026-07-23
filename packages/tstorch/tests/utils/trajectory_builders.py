# pyright: basic
"""ODE-integration trajectory builders backing the cached benchmark/unit-test fixtures.

Kept separate from :mod:`benchmark_utils` so that importing the pure tensor/tsfresh
helpers there does not pull in JAX or the case-study ODE systems.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from jax import Array

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from case_studies.duffing_oscillator.duffing_jax_ode import (  # noqa: E402
    DuffingJaxODE,
    DuffingParams,
)
from case_studies.friction.friction_jax_ode import FrictionJaxODE, FrictionParams  # noqa: E402
from case_studies.lorenz.lorenz_jax_ode import LorenzJaxODE, LorenzParams  # noqa: E402
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams  # noqa: E402
from pybasin.sampler import GridSampler  # noqa: E402
from pybasin.solvers import JaxSolver  # noqa: E402

from .benchmark_utils import (  # noqa: E402
    BenchmarkSeriesData,
    combine_case_data,
    filter_steady,
    grid_sample_count,
)


def integrate_case(
    *,
    name: str,
    ode_system: Any,
    y0: list[list[float]],
    solver: JaxSolver,
    time_steady: float,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    """Integrate one case-study template set and return steady BST data."""
    y0_tensor = torch.tensor(y0, dtype=torch.float32, device=solver.device)
    time, y_tsb = solver.integrate(ode_system, y0_tensor)
    time_steady_tensor, y_steady_tsb = filter_steady(time, y_tsb, time_steady)
    return name, time_steady_tensor, y_steady_tsb.permute(1, 2, 0).contiguous()


def build_canonical_trajectories() -> BenchmarkSeriesData:
    """Build the bounded canonical supervised trajectories."""
    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    pendulum_params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    cases.append(
        integrate_case(
            name="pendulum",
            ode_system=PendulumJaxODE(pendulum_params),
            y0=[
                [0.4, 0.0],
                [2.7, 0.0],
            ],
            solver=JaxSolver(
                t_span=(0, 1000),
                t_steps=1000,
                device="cpu",
                rtol=1e-8,
                atol=1e-6,
                cache_dir=None,
            ),
            time_steady=950.0,
        )
    )

    duffing_params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}
    cases.append(
        integrate_case(
            name="duffing",
            ode_system=DuffingJaxODE(duffing_params),
            y0=[
                [-0.21, 0.02],
                [1.05, 0.77],
                [-0.67, 0.02],
                [-0.46, 0.30],
                [-0.43, 0.12],
            ],
            solver=JaxSolver(
                t_span=(0, 1000),
                t_steps=5000,
                t_eval=(900.0, 1000.0),
                device="cpu",
                rtol=1e-8,
                atol=1e-6,
                cache_dir=None,
            ),
            time_steady=900.0,
        )
    )

    lorenz_params: LorenzParams = {"sigma": 0.12, "r": 0.0, "b": -0.6}
    cases.append(
        integrate_case(
            name="lorenz",
            ode_system=LorenzJaxODE(lorenz_params),
            y0=[
                [0.8, -3.0, 0.0],
                [-0.8, 3.0, 0.0],
            ],
            solver=JaxSolver(
                t_span=(0, 1000),
                t_steps=4000,
                t_eval=(900.0, 1000.0),
                device="cpu",
                rtol=1e-8,
                atol=1e-6,
                cache_dir=None,
                event_fn=lorenz_stop_event,
            ),
            time_steady=900.0,
        )
    )

    friction_params: FrictionParams = {
        "v_d": 1.5,
        "xi": 0.05,
        "musd": 2.0,
        "mud": 0.5,
        "muv": 0.0,
        "v0": 0.5,
    }
    cases.append(
        integrate_case(
            name="friction",
            ode_system=FrictionJaxODE(friction_params),
            y0=[
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            solver=JaxSolver(
                t_span=(0, 500),
                t_steps=500,
                t_eval=(400.0, 500.0),
                device="cpu",
                rtol=1e-8,
                atol=1e-6,
                cache_dir=None,
            ),
            time_steady=400.0,
        )
    )

    data = combine_case_data(cases)
    metadata = {
        **data.metadata,
        "requested_counts": {"pendulum": 2, "duffing": 5, "lorenz": 2, "friction": 2},
        "total_trajectories": data.metadata["num_trajectories"],
        "lorenz_note": "setup_lorenz_system defines 3 templates; benchmark omits the known unbounded template.",
    }
    return BenchmarkSeriesData(
        time=data.time,
        y_bst=data.y_bst,
        case_tensors=data.case_tensors,
        case_times=data.case_times,
        case_slices=data.case_slices,
        metadata=metadata,
    )


def build_pendulum_grid_trajectories() -> BenchmarkSeriesData:
    """Build the 10,000-trajectory pendulum grid dataset."""
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device="cpu",
    )
    y0 = sampler.sample(10_000)
    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=1000,
        device="cpu",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,
    )
    time, y_tsb = solver.integrate(PendulumJaxODE(params), y0)
    time_steady, y_steady_tsb = filter_steady(time, y_tsb, 950.0)
    y_bst = y_steady_tsb.permute(1, 2, 0).contiguous()

    return BenchmarkSeriesData(
        time=time_steady,
        y_bst=y_bst,
        case_tensors={"pendulum_grid": y_bst},
        case_times={"pendulum_grid": time_steady},
        case_slices={"pendulum_grid": (0, y_bst.shape[0])},
        metadata={
            "case": "pendulum",
            "requested_trajectories": 10_000,
            "actual_trajectories": y_bst.shape[0],
            "expected_grid_count": grid_sample_count(10_000, dimensions=2),
            "time_steady": 950.0,
            "solver": {
                "t_span": (0, 1000),
                "t_steps": 1000,
                "rtol": 1e-8,
                "atol": 1e-6,
            },
        },
    )


def lorenz_stop_event(t: Array, y: Array, args: Any, **kwargs: Any) -> Array:
    """Stop Lorenz integration when the trajectory exceeds the setup threshold."""
    del t, args, kwargs
    return 200.0 - jnp.max(jnp.abs(y))
