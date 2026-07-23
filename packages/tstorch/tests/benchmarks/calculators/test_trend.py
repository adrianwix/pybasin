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
FEATURES: list[FeatureSpec] = DOMAIN_FEATURES["trend"]
BATCHED_ARGUMENT_COUNTS: list[int] = [1, 5, 10]


def _feature_params_for_count(feature_name: str, count: int) -> list[dict[str, Any]]:
    if feature_name in {"linear_trend", "linear_trend_timewise"}:
        attributes = ["slope", "intercept", "rvalue", "pvalue", "stderr"]
        return [{"attr": attributes[index % len(attributes)]} for index in range(count)]
    if feature_name == "agg_linear_trend":
        attributes = ["slope", "intercept", "rvalue", "pvalue", "stderr"]
        aggregates = ["mean", "var", "min", "max"]
        return [
            {
                "chunk_size": 5 + 5 * (index // 5),
                "f_agg": aggregates[index % len(aggregates)],
                "attr": attributes[index % len(attributes)],
            }
            for index in range(count)
        ]
    if feature_name == "ar_coefficient":
        return [{"k": 10, "coeff": index} for index in range(count)]
    if feature_name == "augmented_dickey_fuller":
        attributes = ["teststat", "pvalue", "usedlag"]
        autolags: list[str | None] = ["AIC", "BIC", "t-stat", None]
        return [
            {
                "attr": attributes[index % len(attributes)],
                "autolag": autolags[index % len(autolags)],
            }
            for index in range(count)
        ]
    raise ValueError(f"no batched-argument param sweep defined for {feature_name!r}")


def _tsfresh_params(feature_name: str, params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if feature_name != "agg_linear_trend":
        return params
    return [
        {
            "chunk_len": param["chunk_size"],
            "f_agg": param["f_agg"],
            "attr": param["attr"],
        }
        for param in params
    ]


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_single_trajectory_direct_call_uncompiled_trend(
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
def test_single_trajectory_direct_call_compiled_trend(
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
def test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_trend(
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
def test_batched_trajectories_proxy_call_compiled_cold_run_trend(
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
def test_batched_trajectories_proxy_call_compiled_trend(
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
def test_batched_trajectories_proxy_call_uncompiled_trend(
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
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_uncompiled_batched_arguments_trend(
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
        params,
        tsfresh_params=_tsfresh_params(spec.name, params),
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_compiled_batched_arguments_trend(
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
        params,
        tsfresh_params=_tsfresh_params(spec.name, params),
    )
