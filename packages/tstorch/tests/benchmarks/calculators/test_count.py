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
FEATURES: list[FeatureSpec] = DOMAIN_FEATURES["count"]
PARAMETRIZED_FEATURES: list[FeatureSpec] = [
    spec
    for spec in FEATURES
    if spec.name in {"count_above", "count_below", "count_in_range", "count_value"}
]
BATCHED_ARGUMENT_COUNTS: list[int] = [1, 5, 10]


def _feature_params_for_count(feature_name: str, count: int) -> list[dict[str, Any]]:
    if feature_name in {"count_above", "count_below"}:
        return [{"t": float(index - count // 2)} for index in range(count)]
    if feature_name == "count_in_range":
        return [
            {"min_val": float(index - count), "max_val": float(index + 1)} for index in range(count)
        ]
    if feature_name == "count_value":
        return [{"value": float(index - count // 2)} for index in range(count)]
    raise ValueError(f"no batched-argument param sweep defined for {feature_name!r}")


def _params_for_backend(
    feature_name: str,
    params: list[dict[str, Any]],
    backend: Backend,
) -> list[dict[str, Any]]:
    if feature_name == "count_in_range" and backend == "tsfresh":
        return [{"min": param["min_val"], "max": param["max_val"]} for param in params]
    return params


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_single_trajectory_direct_call_uncompiled_count(
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
def test_single_trajectory_direct_call_compiled_count(
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
def test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_count(
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
def test_batched_trajectories_proxy_call_compiled_cold_run_count(
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
def test_batched_trajectories_proxy_call_compiled_count(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_batched_trajectories_proxy_call_compiled_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_proxy_call_uncompiled_count(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    run_batched_trajectories_proxy_call_uncompiled_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


@parametrize_domain(PARAMETRIZED_FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_uncompiled_batched_arguments_count(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    count: int,
) -> None:
    params = _feature_params_for_count(spec.name, count)
    run_batched_trajectories_proxy_call_uncompiled_batched_arguments_benchmark(
        benchmark,
        pendulum_grid_trajectories,
        spec,
        backend,
        _params_for_backend(spec.name, params, backend),
    )


@parametrize_domain(PARAMETRIZED_FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_compiled_batched_arguments_count(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    count: int,
) -> None:
    params = _feature_params_for_count(spec.name, count)
    run_batched_trajectories_proxy_call_compiled_batched_arguments_benchmark(
        benchmark,
        pendulum_grid_trajectories,
        spec,
        backend,
        _params_for_backend(spec.name, params, backend),
    )
