"""Integration tests for the pendulum case study."""

import math
from pathlib import Path

import pytest

from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    run_basin_stability_test,
    run_parameter_study_test,
    run_single_point_test,
)


class TestPendulum:
    """Integration tests for pendulum basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test pendulum baseline against the stored seeded reference data.

        Verifies:
        1. The regression run uses the expected sample count
        2. Classification metrics remain above the acceptance threshold
        """
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
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(
                bse,
                comparison,
                trajectory_state=1,
                trajectory_x_limits=(0, 50 * math.pi),
                trajectory_y_limits={"FP": (-0.12, 0.12), "LC": (0, 6)},
                hidden_state_space_labels=["LC"],
            )

    @pytest.mark.integration
    def test_parameter_t(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test pendulum period (T) sweep against stored seeded reference labels.

        Verifies:
        1. Parameter sweep over T (driving period)
        2. Classification metrics remain above the acceptance threshold at each point
        """
        json_path = Path(__file__).parent / "main_pendulum_case2.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "case2"

        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_pendulum_system,
            parameter_name='ode_system.params["T"]',
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
        """Test hyperparameter n sweep against stored seeded reference labels.

        Verifies:
        1. Parameter sweep over N (sample size)
        2. Classification metrics remain above the acceptance threshold at each point
        """
        json_path = Path(__file__).parent / "main_pendulum_hyperparameters.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperparameters"

        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_pendulum_system,
            parameter_name="n",
            seed=42,
            ground_truths_dir=ground_truths_dir,
            compute_orbit_data=artifact_collector is not None,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(
                bs_study, comparisons, interval="log", skip_orbit_diagram=True
            )

    @pytest.mark.integration
    @pytest.mark.no_artifacts
    def test_n50(self) -> None:
        """Test with small N=50 for random sampling validation.

        Python implementation uses UniformRandomSampler which generates exactly N points.
        Expected basin stability (seed=42, CPU solver):
        - FP: 0.1200, LC: 0.8800
        """
        run_single_point_test(
            n=50,
            expected_bs={"FP": 0.12, "LC": 0.88},
            setup_function=setup_pendulum_system,
            expected_points=50,
            seed=42,
        )
