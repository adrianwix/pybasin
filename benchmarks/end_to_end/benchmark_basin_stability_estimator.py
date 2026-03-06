"""
Pytest-benchmark comparing CPU vs GPU performance for pendulum basin stability estimation.

Benchmarks the complete basin stability workflow across different N values (sample sizes)
to analyze scaling behavior. Uses UniformRandomSampler and PendulumFeatureExtractor
with feature filtering and unbounded detection disabled for controlled comparison.

Hardware:
    CPU: Intel(R) Core(TM) Ultra 9 275HX
    GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU

Run with:
    uv run pytest benchmarks/benchmark_basin_stability_estimator.py --benchmark-only
    uv run pytest benchmarks/benchmark_basin_stability_estimator.py --benchmark-only --benchmark-save=pendulum_scaling

Compare results:
    uv run pytest-benchmark compare
"""

import logging
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.neighbors import KNeighborsClassifier

from case_studies.pendulum.pendulum_feature_extractor import PendulumFeatureExtractor
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.template_integrator import TemplateIntegrator

logging.getLogger("pybasin").setLevel(logging.WARNING)

N_VALUES = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
DEVICES = ["cpu", "cuda"]

PARAMS: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
TIME_SPAN = (0, 1000)
N_STEPS = 1000


def create_bse(n: int, device: str) -> BasinStabilityEstimator:
    print(f"\nRunning: n={n}, device={device}")
    ode_system = PendulumJaxODE(PARAMS)

    sampler = UniformRandomSampler(
        min_limits=[-np.pi + np.arcsin(PARAMS["T"] / PARAMS["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(PARAMS["T"] / PARAMS["K"]), 10.0],
        device=device,
    )

    solver = JaxSolver(
        t_span=TIME_SPAN,
        t_steps=N_STEPS,
        device=device,
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,
    )

    feature_extractor = PendulumFeatureExtractor(
        time_steady=950.0,
    )

    template_y0 = [
        [0.5, 0.0],
        [2.7, 0.0],
    ]
    classifier_labels = ["FP", "LC"]

    knn = KNeighborsClassifier(n_neighbors=1)

    knn = KNeighborsClassifier(n_neighbors=1)

    template_integrator = TemplateIntegrator(
        template_y0=template_y0,
        labels=classifier_labels,
        ode_params=PARAMS,
    )

    return BasinStabilityEstimator(
        n=n,
        ode_system=ode_system,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        predictor=knn,
        template_integrator=template_integrator,
        feature_selector=None,
        detect_unbounded=False,
        output_dir=None,
    )


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("device", DEVICES)
def test_benchmark_n_scaling(benchmark: Any, n: int, device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    bse = create_bse(n, device)

    result = benchmark(bse.estimate_bs)

    assert result is not None
    assert "FP" in result or "LC" in result
