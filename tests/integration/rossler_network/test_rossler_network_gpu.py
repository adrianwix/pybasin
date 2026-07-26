"""GPU-only integration tests for the Rössler network basin stability case study.

Mirrors ``test_rossler_network.py``. ``setup_rossler_network_system`` already runs
on GPU whenever CUDA is available and never forces CPU, so this class reuses the
exact same test bodies -- the only difference is the explicit GPU/cuML requirement,
making this evidence deliberate and gated rather than incidental.
"""

import pytest

from tests.conftest import ArtifactCollector
from tests.gpu.gpu_helpers import requires_gpu_and_cuml
from tests.integration.rossler_network.test_rossler_network import (
    run_rossler_baseline,
    run_rossler_k_sweep,
)


@requires_gpu_and_cuml
@pytest.mark.gpu
class TestRosslerNetworkGPU:
    """GPU-only integration tests for Rössler network basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Rössler network baseline parameters (K=0.218), GPU only.

        Expected from paper:
        - K=0.218: S_B ≈ 0.496

        Verifies:
        1. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3
        2. Results are documented for artifact generation
        """
        run_rossler_baseline(artifact_collector, gpu=True)

    @pytest.mark.integration
    def test_parameter_k(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Rössler network coupling strength (K) parameter sweep, GPU only.

        Studies the effect of varying K from 0.119 to 0.317 (11 values) on basin stability.
        This replicates the K-sweep from the paper.

        Expected from paper: Mean S̄_B ≈ 0.49, monotonically increasing trend

        Verifies:
        1. Parameter sweep over K (coupling strength) executes successfully
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3
        3. Results are documented for artifact generation
        """
        run_rossler_k_sweep(artifact_collector, gpu=True)
