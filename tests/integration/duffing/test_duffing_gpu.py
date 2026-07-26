"""GPU-only integration tests for the Duffing oscillator case study.

Mirrors ``test_duffing.py`` but runs the whole pipeline on the GPU via the real
cuML backend instead of forcing CPU -- evidence that the GPU path produces
results comparable to the CPU-forced regression tests.
"""

from pathlib import Path

import pytest
from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)

from tests.conftest import ArtifactCollector
from tests.gpu.gpu_helpers import requires_gpu_and_cuml
from tests.integration.duffing.test_duffing import run_duffing_unsupervised
from tests.integration.test_helpers import run_basin_stability_test


@requires_gpu_and_cuml
@pytest.mark.gpu
class TestDuffingGPU:
    """GPU-only integration tests for Duffing oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_baseline_supervised(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Duffing oscillator baseline with supervised classification approach, GPU only."""
        json_path = Path(__file__).parent / "main_duffing_supervised.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_duffing.csv"
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_duffing_oscillator_system,
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
                trajectory_state=0,
                trajectory_x_limits=(0, 50),
                trajectory_y_limits=(-1.4, 1.4),
                trajectory_y_ticks=[-1, 1],
                phase_space_axes=(0, 1),
                gpu=True,
            )

    @pytest.mark.integration
    def test_baseline_unsupervised(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Duffing oscillator baseline with unsupervised clustering approach, GPU only."""
        bse, comparison = run_duffing_unsupervised(force_cpu=False)

        if artifact_collector is not None:
            artifact_collector.add_unsupervised(
                bse,
                comparison,
                trajectory_state=0,
                trajectory_x_limits=(0, 50),
                trajectory_y_limits=(-1.4, 1.4),
                trajectory_y_ticks=[-1, 1],
                phase_space_axes=(0, 1),
                gpu=True,
            )
