import pytest

from ...utils.benchmark_utils import (
    run_batched_trajectories_proxy_call_compiled_cold_run_benchmark,
    run_single_trajectory_direct_call_uncompiled_benchmark,
)
from ...utils.domain_runner import BACKENDS, parametrize_domain
from ...utils.feature_specs import DOMAIN_FEATURES

pytestmark = pytest.mark.tstorch_benchmark
FEATURES = DOMAIN_FEATURES["dynamical"]


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_single_trajectory_direct_call_uncompiled_dynamical(
    benchmark, canonical_trajectories, spec, backend
):
    run_single_trajectory_direct_call_uncompiled_benchmark(
        benchmark, canonical_trajectories, spec, backend
    )


@parametrize_domain(FEATURES)
@pytest.mark.parametrize("backend", BACKENDS)
def test_batched_trajectories_proxy_call_compiled_cold_run_dynamical(
    benchmark, pendulum_grid_trajectories, spec, backend
):
    run_batched_trajectories_proxy_call_compiled_cold_run_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend
    )
