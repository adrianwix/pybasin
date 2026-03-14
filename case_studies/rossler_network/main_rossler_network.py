"""Main entry point for the Rössler network basin stability case study.

This replicates the basin stability analysis of synchronization in coupled
Rössler oscillator networks from the original paper.

Expected result for K=0.218: S_B ≈ 0.496 (±0.05 due to sampling variance)
"""

import numpy as np

from case_studies.rossler_network.setup_rossler_network_system import (
    EXPECTED_SB_FROM_PAPER,
    K_VALUES_FROM_PAPER,
    setup_rossler_network_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.types import StudyResult
from pybasin.utils import time_execution


def main() -> tuple[BasinStabilityEstimator, StudyResult]:
    """Run basin stability estimation for Rössler network.

    Uses coupling strength K=0.218 (expected S_B ≈ 0.496 from paper).

    :return: Basin stability estimator with results.
    """
    props = setup_rossler_network_system()

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        output_dir="results",
        feature_selector=None,
        detect_unbounded=False,
    )

    result = bse.run()

    return bse, result


if __name__ == "__main__":
    bse, result = time_execution("main_rossler_network.py", main)

    # Print results and comparison with paper
    k_val = 0.218
    print(f"\nBasin Stability for k={k_val}:")
    print(result["basin_stability"])

    # Find expected value from paper if k is in the list
    k_indices = np.where(np.isclose(K_VALUES_FROM_PAPER, k_val, atol=1e-6))[0]
    if len(k_indices) > 0:
        expected_sb = float(EXPECTED_SB_FROM_PAPER[k_indices[0]])
        print(f"\nExpected S_B (synchronized) from paper: {expected_sb}")
