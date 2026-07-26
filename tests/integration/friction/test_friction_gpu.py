"""GPU-only integration tests for the friction oscillator case study.

Mirrors ``test_friction.py`` but runs the whole pipeline on the GPU via the real
cuML backend instead of forcing CPU -- evidence that the GPU path produces
results comparable to the CPU-forced regression tests.
"""

from pathlib import Path

import pytest
from case_studies.friction.setup_friction_system import setup_friction_system

from tests.conftest import ArtifactCollector
from tests.gpu.gpu_helpers import requires_gpu_and_cuml
from tests.integration.test_helpers import (
    run_basin_stability_test,
    run_parameter_study_test,
)


@requires_gpu_and_cuml
@pytest.mark.gpu
class TestFrictionGPU:
    """GPU-only integration tests for friction oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test friction oscillator baseline parameters, GPU only."""
        json_path = Path(__file__).parent / "main_friction_case1.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_friction.csv"
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_friction_system,
            n=4000,
            seed=42,
            ground_truth_csv=ground_truth_csv,
            coordinate_columns=["x1", "x2"],
            force_cpu=False,
            bs_atol=0.02,
            mcc_threshold=0.9,
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(
                bse,
                comparison,
                trajectory_state=1,
                trajectory_x_limits=(0, 200),
                trajectory_y_limits={"FP": (-1.1, 1.1), "LC": (-2.2, 2.2)},
                gpu=True,
            )

    @pytest.mark.integration
    def test_parameter_v_d(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test friction oscillator v_d parameter sweep, GPU only."""
        json_path = Path(__file__).parent / "main_friction_v_study.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "vStudy"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_friction_system,
            parameter_name='ode_system.params["v_d"]',
            n=2000,
            seed=42,
            ground_truths_dir=ground_truths_dir,
            coordinate_columns=["x1", "x2"],
            compute_orbit_data=artifact_collector is not None,
            force_cpu=False,
            bs_atol=0.02,
            mcc_threshold=0.9,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons, gpu=True)
