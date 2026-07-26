"""GPU-only integration tests for the pendulum case study.

Mirrors ``test_pendulum.py`` but runs the whole pipeline on the GPU via the real
cuML backend instead of forcing CPU -- evidence that the GPU path produces
results comparable to the CPU-forced regression tests.
"""

import math
from pathlib import Path

import pytest
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system

from tests.conftest import ArtifactCollector
from tests.gpu.gpu_helpers import requires_gpu_and_cuml
from tests.integration.test_helpers import (
    run_basin_stability_test,
    run_parameter_study_test,
    run_single_point_test,
)


@requires_gpu_and_cuml
@pytest.mark.gpu
class TestPendulumGPU:
    """GPU-only integration tests for pendulum basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test pendulum baseline against the stored seeded reference data, GPU only."""
        json_path = Path(__file__).parent / "main_pendulum_case1.json"
        ground_truth_csv = (
            Path(__file__).parent / "ground_truths" / "case1" / "main_pendulum_case1.csv"
        )

        bse, comparison = run_basin_stability_test(
            json_path,
            setup_pendulum_system,
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
                trajectory_x_limits=(0, 50 * math.pi),
                trajectory_y_limits={"FP": (-0.12, 0.12), "LC": (0, 6)},
                hidden_state_space_labels=["LC"],
                gpu=True,
            )

    @pytest.mark.integration
    def test_parameter_t(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test pendulum period (T) sweep against stored seeded reference labels, GPU only."""
        json_path = Path(__file__).parent / "main_pendulum_case2.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "case2"

        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_pendulum_system,
            parameter_name='ode_system.params["T"]',
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

    @pytest.mark.integration
    def test_hyperparameter_n(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter n sweep against stored seeded reference labels, GPU only."""
        json_path = Path(__file__).parent / "main_pendulum_hyperparameters.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperparameters"

        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_pendulum_system,
            parameter_name="n",
            seed=42,
            ground_truths_dir=ground_truths_dir,
            coordinate_columns=["x1", "x2"],
            compute_orbit_data=artifact_collector is not None,
            force_cpu=False,
            bs_atol=0.02,
            mcc_threshold=0.9,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(
                bs_study, comparisons, interval="log", skip_orbit_diagram=True, gpu=True
            )

    @pytest.mark.integration
    @pytest.mark.no_artifacts
    def test_n50(self) -> None:
        """Test with small N=50 for random sampling validation, GPU only.

        Python implementation uses UniformRandomSampler which generates exactly N points.
        Expected basin stability (seed=42, GPU solver):
        - FP: 0.1200, LC: 0.8800
        """
        run_single_point_test(
            n=50,
            expected_bs={"FP": 0.12, "LC": 0.88},
            setup_function=setup_pendulum_system,
            expected_points=50,
            seed=42,
            force_cpu=False,
        )
