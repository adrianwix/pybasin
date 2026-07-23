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
FEATURES: list[FeatureSpec] = DOMAIN_FEATURES["statistical"]

# The only statistical features with a genuine param axis - the rest are parameter-free,
# so there is no argument-count sweep to run for them.
PARAMETRIZED_FEATURES: list[FeatureSpec] = [
    spec
    for spec in FEATURES
    if spec.name in {"quantile", "mean_n_absolute_max", "ratio_beyond_r_sigma", "symmetry_looking"}
]


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_single_trajectory_direct_call_uncompiled_statistical(
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
def test_single_trajectory_direct_call_compiled_statistical(
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
def test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_statistical(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    """Direct-call equivalent of test_batched_trajectories_proxy_call_compiled_cold_run_statistical, for features not
    wired into TorchFeatureExtractor (e.g. still-parametrized ones in other domains)."""
    run_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_proxy_call_compiled_cold_run_statistical(
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
def test_batched_trajectories_proxy_call_compiled_statistical(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    """Steady-state cost on ONE reused, already-compiled extractor - see
    run_batched_trajectories_proxy_call_compiled_benchmark for how this differs from
    test_batched_trajectories_proxy_call_compiled_cold_run_statistical (fresh extractor + compile cost every round)."""
    run_batched_trajectories_proxy_call_compiled_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_proxy_call_uncompiled_statistical(
    benchmark: BenchmarkFixture,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> None:
    """Same as test_batched_trajectories_proxy_call_compiled_statistical but with use_compile=False -
    isolates extractor call overhead with no torch.compile step at all."""
    run_batched_trajectories_proxy_call_uncompiled_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )


def _feature_params_for_count(feature_name: str, count: int) -> list[dict[str, Any]]:
    if feature_name == "quantile":
        return [{"q": (i + 1) / (count + 1)} for i in range(count)]
    if feature_name == "mean_n_absolute_max":
        return [{"number_of_maxima": i + 1} for i in range(count)]
    if feature_name == "ratio_beyond_r_sigma":
        return [{"r": 0.25 * (i + 1)} for i in range(count)]
    if feature_name == "symmetry_looking":
        return [{"r": 0.05 * (i + 1)} for i in range(count)]
    raise ValueError(f"no batched-argument param sweep defined for {feature_name!r}")


BATCHED_ARGUMENT_COUNTS: list[int] = [1, 5, 10]


@parametrize_domain(PARAMETRIZED_FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("count", BATCHED_ARGUMENT_COUNTS)
def test_batched_trajectories_proxy_call_uncompiled_batched_arguments_statistical(
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
def test_batched_trajectories_proxy_call_compiled_batched_arguments_statistical(
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
