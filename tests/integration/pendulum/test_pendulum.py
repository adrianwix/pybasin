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
        """Test pendulum baseline using exact MATLAB initial conditions.

        Uses CsvSampler to load the exact ICs from MATLAB bSTAB, eliminating
        sampling variance. Any differences are due to numerical integration
        or feature extraction only.

        Verifies:
        1. Basin stability values match MATLAB within tight tolerance
        2. Z-score validation: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 0.5
        """
        json_path = Path(__file__).parent / "main_pendulum_case1.json"
        ground_truth_csv = (
            Path(__file__).parent / "ground_truths" / "case1" / "main_pendulum_case1.csv"
        )

        bse, comparison = run_basin_stability_test(
            json_path,
            setup_pendulum_system,
            system_name="pendulum",
            case_name="case1",
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
        """Test pendulum period (T) parameter sweep using exact MATLAB initial conditions.

        Uses CsvSampler for each parameter point to load exact ICs from MATLAB bSTAB,
        eliminating sampling variance. Any differences are due to numerical integration
        or feature extraction only.

        Verifies:
        1. Parameter sweep over T (driving period)
        2. Basin stability values match MATLAB using z-score test with z < 2.0
        3. Uses standard errors from both MATLAB (err_FP, err_LC) and Python results
        """
        json_path = Path(__file__).parent / "main_pendulum_case2.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "case2"

        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_pendulum_system,
            parameter_name='ode_system.params["T"]',
            label_keys=["FP", "LC", "NaN"],
            system_name="pendulum",
            case_name="case2",
            ground_truths_dir=ground_truths_dir,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons)

    @pytest.mark.integration
    def test_hyperparameter_n(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test hyperparameter n - convergence study varying sample size N with exact MATLAB ICs.

        Uses CsvSampler for each N value to load exact ICs from MATLAB, eliminating sampling
        variance across different sample sizes.

        Verifies:
        1. Parameter sweep over N (sample size)
        2. Basin stability values match MATLAB using z-score test with z < 2.5
        3. Uses standard errors from both MATLAB (err_FP, err_LC) and Python results
        """
        json_path = Path(__file__).parent / "main_pendulum_hyperparameters.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "hyperparameters"

        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_pendulum_system,
            parameter_name="n",
            label_keys=["FP", "LC", "NaN"],
            system_name="pendulum",
            case_name="case3",
            ground_truths_dir=ground_truths_dir,
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
