"""Integration tests for the Rössler network basin stability case study.

These tests validate the basin stability estimation for coupled Rössler oscillator
networks against aggregate values from Menck et al. (2013), using statistical error
bounds because the publication does not provide per-trajectory labels.
"""

import numpy as np
import pytest
from case_studies.rossler_network.setup_rossler_network_system import (
    EXPECTED_MEAN_SB,
    EXPECTED_SB_FROM_PAPER,
    K_VALUES_FROM_PAPER,
    setup_rossler_network_system,
)
from test_support import (
    CPU_AND_GPU_TARGETS,
    BasinFractionComparison,
    ExecutionTarget,
    compute_statistical_comparison,
)

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.study_params import SweepStudyParams
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import ComparisonResult


def run_rossler_baseline(
    artifact_collector: ArtifactCollector | None, *, execution_target: ExecutionTarget
) -> None:
    """Run the Rössler network K=0.218 baseline against the paper's expected S_B.

    :param execution_target: Device for sampling, integration, and feature extraction.
    """
    k_val = 0.218
    k_idx = np.where(k_val == K_VALUES_FROM_PAPER)[0][0]
    expected_sb = float(EXPECTED_SB_FROM_PAPER[k_idx])

    props = setup_rossler_network_system(device=execution_target.value)

    bse = BasinStabilityEstimator(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=props.get("solver"),
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        feature_selector=None,
    )

    result = bse.run()
    basin_stability = result["basin_stability"]

    # Get individual attractor basin stabilities
    sync_bs = basin_stability.get("synchronized", 0.0)
    unbounded_bs = basin_stability.get("unbounded", 0.0)

    # For comparison with paper, we use total bounded (synchronized + desynchronized if present)
    computed_sb = sync_bs + basin_stability.get("desynchronized", 0.0)

    n_samples = props["n"]

    # The paper reports the total bounded fraction, which includes synchronized and
    # desynchronized bounded trajectories.
    e_abs_computed = np.sqrt(computed_sb * (1 - computed_sb) / n_samples)
    e_abs_paper = np.sqrt(expected_sb * (1 - expected_sb) / n_samples)
    e_rel_computed = 1.0 / np.sqrt(n_samples * computed_sb)
    e_rel_paper = 1.0 / np.sqrt(n_samples * expected_sb)

    stats_comp = compute_statistical_comparison(
        computed_sb, e_abs_computed, expected_sb, e_abs_paper
    )

    print(f"\nRössler Network K={k_val}:")
    print(f"  Expected S_B: {expected_sb:.3f} ± {e_abs_paper:.3f}")
    print(f"  Computed S_B: {computed_sb:.3f} ± {e_abs_computed:.3f}")
    print(f"  - Synchronized: {sync_bs:.3f}")
    print(f"  - Unbounded: {unbounded_bs:.3f}")
    print(f"  Difference:   {abs(computed_sb - expected_sb):+.3f}")
    print(f"  Z-score:      {stats_comp.z_score:.2f}")
    z_threshold = 3.0
    print(f"  Within {z_threshold:.1f} standard errors: {stats_comp.z_score < z_threshold}")

    # Create comparisons for both attractors
    attractor_comparisons: list[BasinFractionComparison] = []

    attractor_comparisons.append(
        BasinFractionComparison(
            label="bounded",
            actual_bs=computed_sb,
            actual_e_abs=float(e_abs_computed),
            actual_e_rel=float(e_rel_computed),
            reference_bs=expected_sb,
            reference_e_abs=float(e_abs_paper),
            reference_e_rel=float(e_rel_paper),
        )
    )

    reference_unbounded = 1.0 - expected_sb
    e_rel_unbounded = 1.0 / np.sqrt(n_samples * unbounded_bs)
    e_rel_reference_unbounded = 1.0 / np.sqrt(n_samples * reference_unbounded)
    attractor_comparisons.append(
        BasinFractionComparison(
            label="unbounded",
            actual_bs=unbounded_bs,
            actual_e_abs=float(e_abs_computed),
            actual_e_rel=float(e_rel_unbounded),
            reference_bs=reference_unbounded,
            reference_e_abs=float(e_abs_paper),
            reference_e_rel=float(e_rel_reference_unbounded),
        )
    )

    comparison_result = ComparisonResult(
        system_name="rossler_network",
        case_name="baseline",
        attractors=attractor_comparisons,
        reference_name="Menck et al. (2013)",
        uncertainty_sigma_limit=z_threshold,
        paper_validation=True,
    )

    assert stats_comp.z_score < z_threshold, (
        f"Basin stability for K={k_val}: expected {expected_sb:.3f} ± {e_abs_paper:.3f}, "
        f"got {computed_sb:.3f} ± {e_abs_computed:.3f}, "
        f"z-score {stats_comp.z_score:.2f} exceeds threshold {z_threshold:.1f}"
    )

    if artifact_collector is not None:
        artifact_collector.add_single_point(bse, comparison_result, gpu=execution_target.is_gpu)


def run_rossler_k_sweep(
    artifact_collector: ArtifactCollector | None, *, execution_target: ExecutionTarget
) -> None:
    """Run the Rössler network K-sweep against the paper's expected S_B curve.

    :param execution_target: Device for sampling, integration, and feature extraction.
    """
    props = setup_rossler_network_system(device=execution_target.value)

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

    bs_study = BasinStabilityStudy(
        n=props["n"],
        ode_system=props["ode_system"],
        sampler=props["sampler"],
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
    )

    bs_study.run()

    comparison_results: list[ComparisonResult] = []
    computed_sync: list[float] = []
    n_samples = props["n"]
    z_threshold = 3.0

    print("\n" + "=" * 80)
    print("RÖSSLER NETWORK K-SWEEP: COMPARISON WITH PAPER")
    print("=" * 80)
    print(
        f"{'K':>8} | {'Expected':>8} | {'Computed':>8} | {'Diff':>7} | {'e_abs':>6} | {'Z-score':>8} | {'Status':>6}"
    )
    print("-" * 80)

    within_threshold = 0
    for idx, result in enumerate(bs_study.results):
        study_label = result["study_label"]
        bs_dict = result["basin_stability"]
        assert isinstance(study_label, dict)
        param_val = study_label["K"]
        sync_val = bs_dict.get("synchronized", 0.0)
        unbounded_val = bs_dict.get("unbounded", 0.0)

        # For comparison with paper, sum synchronized + desynchronized (if present)
        total_bounded = sync_val + bs_dict.get("desynchronized", 0.0)
        computed_sync.append(total_bounded)
        expected_val = float(EXPECTED_SB_FROM_PAPER[idx])
        e_abs_computed = np.sqrt(total_bounded * (1 - total_bounded) / n_samples)
        e_abs_paper = np.sqrt(expected_val * (1 - expected_val) / n_samples)
        e_rel_computed = 1.0 / np.sqrt(n_samples * total_bounded)
        e_rel_paper = 1.0 / np.sqrt(n_samples * expected_val)

        stats_comp = compute_statistical_comparison(
            total_bounded, e_abs_computed, expected_val, e_abs_paper
        )

        diff = total_bounded - expected_val

        within = stats_comp.z_score < z_threshold
        within_threshold += int(within)
        status = "✓" if within else "✗"

        print(
            f"{param_val:>8.3f} | {expected_val:>8.3f} | {total_bounded:>8.3f} | "
            f"{diff:>+7.3f} | {e_abs_computed:>6.3f} | {stats_comp.z_score:>8.2f} | {status:>6}"
        )

        # Create comparisons for both attractors
        attractor_comparisons_for_param: list[BasinFractionComparison] = []

        attractor_comparisons_for_param.append(
            BasinFractionComparison(
                label="bounded",
                actual_bs=total_bounded,
                actual_e_abs=float(e_abs_computed),
                actual_e_rel=float(e_rel_computed),
                reference_bs=expected_val,
                reference_e_abs=float(e_abs_paper),
                reference_e_rel=float(e_rel_paper),
            )
        )

        reference_unbounded = 1.0 - expected_val
        e_rel_unbounded = 1.0 / np.sqrt(n_samples * unbounded_val)
        e_rel_reference_unbounded = 1.0 / np.sqrt(n_samples * reference_unbounded)
        attractor_comparisons_for_param.append(
            BasinFractionComparison(
                label="unbounded",
                actual_bs=unbounded_val,
                actual_e_abs=float(e_abs_computed),
                actual_e_rel=float(e_rel_unbounded),
                reference_bs=reference_unbounded,
                reference_e_abs=float(e_abs_paper),
                reference_e_rel=float(e_rel_reference_unbounded),
            )
        )

        comparison_result = ComparisonResult(
            system_name="rossler_network",
            case_name="k_sweep",
            attractors=attractor_comparisons_for_param,
            parameter_value=param_val,
            reference_name="Menck et al. (2013)",
            uncertainty_sigma_limit=z_threshold,
            paper_validation=True,
        )
        comparison_results.append(comparison_result)

        assert stats_comp.z_score < z_threshold, (
            f"Basin stability for K={param_val}: expected {expected_val:.3f} ± {e_abs_paper:.3f}, "
            f"got {total_bounded:.3f} ± {e_abs_computed:.3f}, "
            f"z-score {stats_comp.z_score:.2f} exceeds threshold {z_threshold:.1f}"
        )

    mean_sb = np.mean(computed_sync)
    print("-" * 80)
    print(f"Mean S_B: {mean_sb:.3f} (expected ~{EXPECTED_MEAN_SB:.3f} from paper)")
    print(
        f"\nWithin threshold: {within_threshold}/{len(K_VALUES_FROM_PAPER)} "
        f"({within_threshold / len(K_VALUES_FROM_PAPER) * 100:.0f}%)"
    )
    print(f"Note: e_abs = sqrt(S_B*(1-S_B)/N), N={n_samples}, z-threshold={z_threshold:.1f}")

    if artifact_collector is not None:
        artifact_collector.add_parameter_sweep(
            bs_study, comparison_results, gpu=execution_target.is_gpu
        )


@pytest.mark.parametrize("execution_target", CPU_AND_GPU_TARGETS)
class TestRosslerNetwork:
    """Integration tests for Rössler network basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
        execution_target: ExecutionTarget,
    ) -> None:
        """Test Rössler network baseline parameters (K=0.218).

        Expected from paper:
        - K=0.218: S_B ≈ 0.496

        Verifies:
        1. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3
        2. Results are documented for artifact generation
        """
        run_rossler_baseline(artifact_collector, execution_target=execution_target)

    @pytest.mark.integration
    def test_parameter_k(
        self,
        artifact_collector: ArtifactCollector | None,
        execution_target: ExecutionTarget,
    ) -> None:
        """Test Rössler network coupling strength (K) parameter sweep.

        Studies the effect of varying K from 0.119 to 0.317 (11 values) on basin stability.
        This replicates the K-sweep from the paper.

        Expected from paper: Mean S̄_B ≈ 0.49, monotonically increasing trend

        Verifies:
        1. Parameter sweep over K (coupling strength) executes successfully
        2. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3
        3. Results are documented for artifact generation
        """
        run_rossler_k_sweep(artifact_collector, execution_target=execution_target)
