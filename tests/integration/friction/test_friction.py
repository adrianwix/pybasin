"""Integration tests for the friction oscillator case study."""

from pathlib import Path

import pytest

from case_studies.friction.setup_friction_system import setup_friction_system
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    run_basin_stability_test,
    run_parameter_study_test,
)


class TestFriction:
    """Integration tests for friction oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test friction oscillator baseline parameters.

        Parameters: v_d=1.5, ξ=0.05, μsd=2.0, μd=0.5, μv=0.0, v0=0.5
        Expected attractors: FP (fixed point), LC (limit cycle)

        Verifies:
        1. Number of ICs used matches the stored reference count
        2. Classification metrics: MCC >= 0.95
        """
        json_path = Path(__file__).parent / "main_friction_case1.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_friction.csv"
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_friction_system,
            n=4000,
            seed=42,
            ground_truth_csv=ground_truth_csv,
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(
                bse,
                comparison,
                trajectory_state=1,
                trajectory_x_limits=(0, 200),
                trajectory_y_limits={"FP": (-1.1, 1.1), "LC": (-2.2, 2.2)},
            )

    @pytest.mark.integration
    def test_parameter_v_d(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test friction oscillator v_d parameter sweep - parameter study varying v_d.

        Studies the effect of varying driving velocity v_d from 1.85 to 2.0.

        Verifies:
        1. Parameter sweep over v_d (driving velocity)
        2. Classification metrics remain above the acceptance threshold
        """
        json_path = Path(__file__).parent / "main_friction_v_study.json"
        ground_truths_dir = Path(__file__).parent / "ground_truths" / "vStudy"
        bs_study, comparisons = run_parameter_study_test(
            json_path,
            setup_friction_system,
            parameter_name='ode_system.params["v_d"]',
            n=2000,
            seed=42,
            ground_truths_dir=ground_truths_dir,
            compute_orbit_data=artifact_collector is not None,
        )

        if artifact_collector is not None:
            artifact_collector.add_parameter_sweep(bs_study, comparisons)
