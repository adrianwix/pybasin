"""Parameter study for Rössler network basin stability across K values.

This replicates the full K-sweep from the original paper, computing basin stability
at 11 equally spaced values of K in the stability interval.

Expected results from paper:
    K=0.119: S_B=0.226    K=0.238: S_B=0.594
    K=0.139: S_B=0.274    K=0.258: S_B=0.628
    K=0.159: S_B=0.330    K=0.278: S_B=0.656
    K=0.179: S_B=0.346    K=0.297: S_B=0.694
    K=0.198: S_B=0.472    K=0.317: S_B=0.690
    K=0.218: S_B=0.496

Mean basin stability: S̄_B ≈ 0.49
"""

import numpy as np

from case_studies.rossler_network.setup_rossler_network_system import (
    EXPECTED_MEAN_SB,
    EXPECTED_SB_FROM_PAPER,
    K_VALUES_FROM_PAPER,
    setup_rossler_network_system,
)
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.study_params import SweepStudyParams
from pybasin.utils import time_execution


def main() -> BasinStabilityStudy:
    """Run parameter study for Rössler network coupling strength.

    Sweeps through K values from paper to analyze basin stability variation.

    :return: Basin stability study with results.
    """
    props = setup_rossler_network_system()

    study_params = SweepStudyParams(
        **{'ode_system.params["K"]': K_VALUES_FROM_PAPER.tolist()},
    )

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    estimator = props.get("estimator")
    template_integrator = props.get("template_integrator")
    assert solver is not None
    assert feature_extractor is not None
    assert estimator is not None

    bse = BasinStabilityStudy(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
        output_dir="results_k_study",
    )

    bse.run()

    return bse


if __name__ == "__main__":
    bse = time_execution("main_rossler_network_k_study.py", main)

    # Print comparison with paper results
    print("\n" + "=" * 60)
    print("COMPARISON WITH PAPER RESULTS")
    print("=" * 60)

    n_samples = bse.n

    print(f"{'K':>8} | {'Expected':>8} | {'Computed':>8} | {'Diff':>7} | {'e_abs':>6} | {'2σ':>6}")
    print("-" * 60)

    computed_sync: list[float] = []
    within_2sigma = 0
    for idx, result in enumerate(bse.results):
        study_label = result["study_label"]
        bs_dict = result["basin_stability"]
        sync_val = bs_dict.get("synchronized", 0.0)
        computed_sync.append(sync_val)
        expected_val = float(EXPECTED_SB_FROM_PAPER[idx])
        diff = sync_val - expected_val
        assert isinstance(study_label, dict)
        param_val = study_label["K"]

        e_abs = np.sqrt(sync_val * (1 - sync_val) / n_samples)
        e_combined = e_abs * np.sqrt(2)
        within = abs(diff) <= 2 * e_combined
        within_2sigma += int(within)
        status = "✓" if within else "✗"

        print(
            f"{param_val:>8.3f} | {expected_val:>8.3f} | {sync_val:>8.3f} | {diff:>+7.3f} | {e_abs:>6.3f} | {status:>6}"
        )

    if computed_sync:
        mean_sb = np.mean(computed_sync)
        e_mean = np.std(computed_sync, ddof=1) / np.sqrt(len(computed_sync))
        mean_diff = mean_sb - EXPECTED_MEAN_SB
        mean_within = abs(mean_diff) <= 2 * e_mean
        mean_status = "✓" if mean_within else "✗"
        print("-" * 60)
        print(
            f"{'Mean S_B':>8} | {EXPECTED_MEAN_SB:>8.3f} | {mean_sb:>8.3f} | {mean_diff:>+7.3f} | {e_mean:>6.3f} | {mean_status:>6}"
        )
        print(
            f"\nWithin 2σ: {within_2sigma}/{len(K_VALUES_FROM_PAPER)} ({within_2sigma / len(K_VALUES_FROM_PAPER) * 100:.0f}%)"
        )
        print(f"Note: e_abs = sqrt(S_B*(1-S_B)/N), N={n_samples}; e_mean = std(S_B)/sqrt(K))")
