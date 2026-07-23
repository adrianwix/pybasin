from __future__ import annotations

from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from ...utils.benchmark_utils import (
    Backend,
    BenchmarkSeriesData,
    FeatureSpec,
    run_batched_trajectories_proxy_call_compiled_batched_arguments_benchmark,
    run_batched_trajectories_proxy_call_compiled_benchmark,
    run_batched_trajectories_proxy_call_compiled_cold_run_benchmark,
    run_batched_trajectories_proxy_call_uncompiled_batched_arguments_benchmark,
    run_batched_trajectories_proxy_call_uncompiled_benchmark,
    run_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_benchmark,
    run_single_trajectory_direct_call_compiled_benchmark,
    run_single_trajectory_direct_call_uncompiled_benchmark,
)
from ...utils.domain_runner import BACKENDS, parametrize_domain
from ...utils.feature_specs import DOMAIN_FEATURES

pytestmark = pytest.mark.tstorch_benchmark

FEATURES: list[FeatureSpec] = DOMAIN_FEATURES["entropy_complexity"]
PARAMETRIZED_FEATURES: list[FeatureSpec] = [
    spec
    for spec in FEATURES
    if spec.name
    in {
        "permutation_entropy",
        "binned_entropy",
        "fourier_entropy",
        "lempel_ziv_complexity",
        "approximate_entropy",
    }
]
BATCHED_ARGUMENT_COUNTS: list[int] = [1, 5, 10]


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_single_trajectory_direct_call_uncompiled_entropy_complexity(
    benchmark: BenchmarkFixture,
    canonical_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_single_trajectory_direct_call_uncompiled_benchmark(
        benchmark, canonical_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_single_trajectory_direct_call_compiled_entropy_complexity(
    benchmark: BenchmarkFixture,
    canonical_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_single_trajectory_direct_call_compiled_benchmark(
        benchmark, canonical_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_entropy_complexity(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_proxy_call_compiled_cold_run_entropy_complexity(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_batched_trajectories_proxy_call_compiled_cold_run_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_proxy_call_uncompiled_entropy_complexity(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_batched_trajectories_proxy_call_uncompiled_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_proxy_call_compiled_entropy_complexity(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_batched_trajectories_proxy_call_compiled_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


def _feature_params_for_count(feature_name: str, count: int) -> list[dict[str, Any]]:
    if feature_name == "permutation_entropy":
        return [{"tau": 1 + index // 4, "dimension": 2 + index % 4} for index in range(count)]
    if feature_name == "binned_entropy":
        return [{"max_bins": index + 2} for index in range(count)]
    if feature_name == "fourier_entropy":
        return [{"bins": index + 2} for index in range(count)]
    if feature_name == "lempel_ziv_complexity":
        return [{"bins": index + 2} for index in range(count)]
    if feature_name == "approximate_entropy":
        return [{"m": 2, "r": 0.1 * (index + 1)} for index in range(count)]
    raise ValueError(f"no batched-argument param sweep defined for {feature_name!r}")


@parametrize_domain(PARAMETRIZED_FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_uncompiled_batched_arguments_entropy_complexity(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    count: int,
) -> None:
    params = _feature_params_for_count(spec.name, count)
    run_batched_trajectories_proxy_call_uncompiled_batched_arguments_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend, params
    )


@parametrize_domain(PARAMETRIZED_FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_compiled_batched_arguments_entropy_complexity(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    count: int,
) -> None:
    params = _feature_params_for_count(spec.name, count)
    run_batched_trajectories_proxy_call_compiled_batched_arguments_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend, params
    )
