"""GPU-only integration tests for the Lorenz system case study.

Mirrors ``test_lorenz.py`` but runs the whole pipeline on the GPU via the real
cuML backend instead of forcing CPU -- evidence that the GPU path produces
results comparable to the CPU-forced regression tests.
"""

from pathlib import Path

import pytest
from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system

from tests.conftest import ArtifactCollector
from tests.gpu.gpu_helpers import requires_gpu_and_cuml
from tests.integration.test_helpers import (
    run_basin_stability_test,
    run_parameter_study_test,
    run_single_point_test,
)


@requires_gpu_and_cuml
@pytest.mark.gpu
class TestLorenzGPU:
    """GPU-only integration tests for Lorenz system basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz system case 1 - broken butterfly attractor parameters, GPU only."""
        json_path = Path(__file__).parent / "main_lorenz.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_lorenz.csv"
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_lorenz_system,
            n=4000,
            seed=42,
            ground_truth_csv=ground_truth_csv,
            coordinate_columns=["x1", "x2", "x3"],
            force_cpu=False,
            bs_atol=0.02,
            mcc_threshold=0.9,
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(
                bse,
                comparison,
                phase_space_axes=(0, 2),
                hidden_state_space_labels=["unbounded"],
                hidden_phase_space_labels=["unbounded"],
                gpu=True,
            )

    @pytest.mark.integration
    def test_parameter_sigma(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz sigma parameter sweep using classification metrics, GPU only."""
        json_path = Path(__file__).parent / "main_lorenz_sigma_study.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "sigmaStudy"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_lorenz_system,
            parameter_name='ode_system.params["sigma"]',
            n=2000,
            seed=42,
            ground_truths_dir=ground_truths_dir,
            coordinate_columns=["x1", "x2", "x3"],
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
        """Test hyperparameter n - convergence study varying sample size N, GPU only."""
        json_path = Path(__file__).parent / "main_lorenz_hyperparameters.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperpN"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_lorenz_system,
            parameter_name="n",
            seed=42,
            ground_truths_dir=ground_truths_dir,
            coordinate_columns=["x1", "x2", "x3"],
            compute_orbit_data=artifact_collector is not None,
            force_cpu=False,
            bs_atol=0.02,
            mcc_threshold=0.9,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons, gpu=True)

    @pytest.mark.integration
    @pytest.mark.no_artifacts
    def test_n200(self) -> None:
        """Test with small N=200 for random sampling validation, GPU only.

        Expected basin stability (seed=42, GPU solver):
        - chaotic attractor 1: 0.0800, chaotic attractor 2: 0.0700, unbounded: 0.8500
        """
        run_single_point_test(
            n=200,
            expected_bs={
                "chaotic attractor 1": 0.08,
                "chaotic attractor 2": 0.07,
                "unbounded": 0.85,
            },
            setup_function=setup_lorenz_system,
            expected_points=200,
            seed=42,
            force_cpu=False,
        )

    @pytest.mark.integration
    def test_hyperparameter_rtol(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter rtol - ODE solver relative tolerance convergence study, GPU only."""
        json_path = Path(__file__).parent / "main_lorenz_hyperpTol.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperpTol"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_lorenz_system,
            parameter_name="solver.rtol",
            n=2000,
            seed=42,
            ground_truths_dir=ground_truths_dir,
            coordinate_columns=["x1", "x2", "x3"],
            compute_orbit_data=artifact_collector is not None,
            force_cpu=False,
            bs_atol=0.02,
            mcc_threshold=0.4,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons, interval="log", gpu=True)
