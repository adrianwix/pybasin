"""Integration tests for the Lorenz system case study."""

from pathlib import Path

import pytest

from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    run_basin_stability_test,
    run_parameter_study_test,
    run_single_point_test,
)


class TestLorenz:
    """Integration tests for Lorenz system basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz system case 1 - broken butterfly attractor parameters.

        Parameters: sigma=0.12, r=0.0, b=-0.6
        Expected attractors: chaotic attractor 1, chaotic attractor 2, unbounded

        Verifies:
        1. Number of ICs used matches the stored reference count
        2. Classification metrics: MCC >= 0.95
        """
        json_path = Path(__file__).parent / "main_lorenz.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_lorenz.csv"
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_lorenz_system,
            n=4000,
            seed=42,
            ground_truth_csv=ground_truth_csv,
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(
                bse,
                comparison,
                phase_space_axes=(0, 2),
                hidden_state_space_labels=["unbounded"],
                hidden_phase_space_labels=["unbounded"],
            )

    @pytest.mark.integration
    def test_parameter_sigma(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Lorenz sigma parameter sweep using classification metrics.

        Studies the effect of varying sigma parameter from 0.12 to 0.18 on basin stability.

        Verifies:
        1. Parameter sweep over sigma
        2. Classification metrics: MCC >= 0.95 for each parameter point
        """
        json_path = Path(__file__).parent / "main_lorenz_sigma_study.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "sigmaStudy"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_lorenz_system,
            parameter_name='ode_system.params["sigma"]',
            n=2000,
            seed=42,
            ground_truths_dir=ground_truths_dir,
            compute_orbit_data=artifact_collector is not None,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons)

    @pytest.mark.integration
    def test_hyperparameter_n(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter n - convergence study varying sample size N.

        Verifies:
        1. Parameter sweep over N (sample size)
        2. Classification metrics: MCC >= 0.95 for each parameter point
        """
        json_path = Path(__file__).parent / "main_lorenz_hyperparameters.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperpN"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_lorenz_system,
            parameter_name="n",
            seed=42,
            ground_truths_dir=ground_truths_dir,
            compute_orbit_data=artifact_collector is not None,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons)

    @pytest.mark.integration
    @pytest.mark.no_artifacts
    def test_n200(self) -> None:
        """Test with small N=200 for random sampling validation.

        Expected basin stability (seed=42, CPU solver):
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
        )

    @pytest.mark.integration
    def test_hyperparameter_rtol(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter rtol - ODE solver relative tolerance convergence study.

        Studies the effect of varying relative tolerance from 1e-3 to 1e-8 on basin stability.
        This test validates that the solver correctly uses the specified tolerances using
        classification metrics to compare against ground truth labels.

        Expected behavior:
        - rtol=1e-3: May produce different results (tolerance is too coarse)
        - rtol=1e-4 to 1e-8: Should converge to consistent classification results
        """
        json_path = Path(__file__).parent / "main_lorenz_hyperpTol.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperpTol"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_lorenz_system,
            parameter_name="solver.rtol",
            n=2000,
            seed=42,
            ground_truths_dir=ground_truths_dir,
            mcc_threshold=0.4,
            compute_orbit_data=artifact_collector is not None,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons, interval="log")
