"""Integration tests for the Rössler network basin stability case study.

These tests validate the basin stability estimation for coupled Rössler oscillator
networks against expected values from the reference paper (Menck et al., 2013).

Note: Unlike other case studies, this does not have MATLAB bSTAB reference implementation,
so we validate against the published paper results using statistical error bounds.
"""

import numpy as np
import pytest

from case_studies.rossler_network.setup_rossler_network_system import (
    EXPECTED_MEAN_SB,
    EXPECTED_SB_FROM_PAPER,
    K_VALUES_FROM_PAPER,
    setup_rossler_network_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.study_params import SweepStudyParams
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    AttractorComparison,
    ComparisonResult,
    compute_statistical_comparison,
)


class TestRosslerNetwork:
    """Integration tests for Rössler network basin stability estimation."""

    @pytest.mark.integration
    def test_baseline(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Rössler network baseline parameters (K=0.218).

        Expected from paper:
        - K=0.218: S_B ≈ 0.496

        Verifies:
        1. Basin stability values pass z-score test: z = |A-B|/sqrt(SE_A^2 + SE_B^2) < 3
        2. Results are documented for artifact generation
        """
        k_val = 0.218
        k_idx = np.where(k_val == K_VALUES_FROM_PAPER)[0][0]
        expected_sb = float(EXPECTED_SB_FROM_PAPER[k_idx])

        props = setup_rossler_network_system()

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

        result = bse.estimate_bs()
        basin_stability = result["basin_stability"]

        # Get individual attractor basin stabilities
        sync_bs = basin_stability.get("synchronized", 0.0)
        unbounded_bs = basin_stability.get("unbounded", 0.0)

        # For comparison with paper, we use total bounded (synchronized + desynchronized if present)
        computed_sb = sync_bs + basin_stability.get("desynchronized", 0.0)

        n_samples = props["n"]

        # Compute standard errors for individual attractors
        sync_se = np.sqrt(sync_bs * (1 - sync_bs) / n_samples)
        unbounded_se = np.sqrt(unbounded_bs * (1 - unbounded_bs) / n_samples)

        # Standard errors for comparison with paper
        e_abs_computed = np.sqrt(computed_sb * (1 - computed_sb) / n_samples)
        e_abs_paper = np.sqrt(expected_sb * (1 - expected_sb) / n_samples)

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
        print(f"  Passes test:  {stats_comp.passes_test} (α=0.05)")

        # Note: For comparison with old code, 1.96 corresponds to α=0.05 (95% CI)
        # Old threshold of 3.0 corresponds to α=0.0027 (99.7% CI)
        z_threshold = 3.0  # Using stricter threshold for paper validation

        # Create comparisons for both attractors
        attractor_comparisons: list[AttractorComparison] = []

        # Synchronized attractor
        attractor_comparisons.append(
            AttractorComparison(
                label="synchronized",
                python_bs=sync_bs,
                python_se=sync_se,
                matlab_bs=expected_sb,  # Paper reports total bounded
                matlab_se=e_abs_paper,
            )
        )

        # Unbounded attractor
        attractor_comparisons.append(
            AttractorComparison(
                label="unbounded",
                python_bs=unbounded_bs,
                python_se=unbounded_se,
                matlab_bs=1.0 - expected_sb,  # Everything not synchronized
                matlab_se=e_abs_paper,
            )
        )

        comparison_result = ComparisonResult(
            system_name="rossler_network",
            case_name="baseline",
            attractors=attractor_comparisons,
            paper_validation=True,
        )

        assert stats_comp.z_score < z_threshold, (
            f"Basin stability for K={k_val}: expected {expected_sb:.3f} ± {e_abs_paper:.3f}, "
            f"got {computed_sb:.3f} ± {e_abs_computed:.3f}, "
            f"z-score {stats_comp.z_score:.2f} exceeds threshold {z_threshold:.1f}"
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(bse, comparison_result)

    @pytest.mark.integration
    def test_parameter_k(
        self,
        artifact_collector: ArtifactCollector | None,
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
        z_threshold = 3.0  # Using stricter threshold for paper validation

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

            # Compute SE for individual attractors
            sync_se = np.sqrt(sync_val * (1 - sync_val) / n_samples)
            unbounded_se = np.sqrt(unbounded_val * (1 - unbounded_val) / n_samples)

            # For comparison with paper, sum synchronized + desynchronized (if present)
            total_bounded = sync_val + bs_dict.get("desynchronized", 0.0)
            computed_sync.append(total_bounded)
            expected_val = float(EXPECTED_SB_FROM_PAPER[idx])
            e_abs_computed = np.sqrt(total_bounded * (1 - total_bounded) / n_samples)
            e_abs_paper = np.sqrt(expected_val * (1 - expected_val) / n_samples)

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
            attractor_comparisons_for_param: list[AttractorComparison] = []

            # Synchronized attractor
            attractor_comparisons_for_param.append(
                AttractorComparison(
                    label="synchronized",
                    python_bs=sync_val,
                    python_se=sync_se,
                    matlab_bs=expected_val,  # Paper reports total bounded
                    matlab_se=e_abs_paper,
                )
            )

            # Unbounded attractor
            attractor_comparisons_for_param.append(
                AttractorComparison(
                    label="unbounded",
                    python_bs=unbounded_val,
                    python_se=unbounded_se,
                    matlab_bs=1.0 - expected_val,  # Everything not synchronized
                    matlab_se=e_abs_paper,
                )
            )

            comparison_result = ComparisonResult(
                system_name="rossler_network",
                case_name="k_sweep",
                attractors=attractor_comparisons_for_param,
                parameter_value=param_val,
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
            artifact_collector.add_parameter_sweep(bs_study, comparison_results)
