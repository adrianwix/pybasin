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
FEATURES: list[FeatureSpec] = DOMAIN_FEATURES["frequency"]
PARAMETRIZED_FEATURES: list[FeatureSpec] = [
    spec
    for spec in FEATURES
    if spec.name in {"fft_coefficient", "spkt_welch_density", "cwt_coefficients"}
]
BATCHED_ARGUMENT_COUNTS: list[int] = [1, 5, 10]


def _feature_params_for_count(feature_name: str, count: int) -> list[dict[str, Any]]:
    if feature_name == "fft_coefficient":
        attributes = ("real", "imag", "abs", "angle")
        return [
            {"coeff": index, "attr": attributes[index % len(attributes)]} for index in range(count)
        ]
    if feature_name == "spkt_welch_density":
        return [{"coeff": index} for index in range(count)]
    if feature_name == "cwt_coefficients":
        return [{"widths": (index + 1,), "coeff": index, "w": index + 1} for index in range(count)]
    raise ValueError(f"no batched-argument param sweep defined for {feature_name!r}")


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_single_trajectory_direct_call_uncompiled_frequency(
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
def test_single_trajectory_direct_call_compiled_frequency(
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
def test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_frequency(
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
def test_batched_trajectories_proxy_call_compiled_cold_run_frequency(
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
def test_batched_trajectories_proxy_call_compiled_frequency(
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
def test_batched_trajectories_proxy_call_uncompiled_frequency(
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
def test_batched_trajectories_proxy_call_uncompiled_batched_arguments_frequency(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    count: int,
) -> None:
    run_batched_trajectories_proxy_call_uncompiled_batched_arguments_benchmark(
        benchmark,
        pendulum_grid_trajectories,
        spec,
        backend,
        _feature_params_for_count(spec.name, count),
    )


@parametrize_domain(PARAMETRIZED_FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_compiled_batched_arguments_frequency(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    count: int,
) -> None:
    run_batched_trajectories_proxy_call_compiled_batched_arguments_benchmark(
        benchmark,
        pendulum_grid_trajectories,
        spec,
        backend,
        _feature_params_for_count(spec.name, count),
    )
