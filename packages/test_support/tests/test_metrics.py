import math

import numpy as np
import pytest
from test_support import (
    compute_basin_fraction_comparisons,
    compute_classification_metrics,
    compute_statistical_comparison,
)


class TestClassificationMetrics:
    def test_matches_hand_computed_values(self) -> None:
        metrics = compute_classification_metrics(
            np.array(["a", "a", "b", "b"]), np.array(["a", "a", "b", "a"])
        )

        assert metrics.f1_per_class == pytest.approx({"a": 4 / 5, "b": 2 / 3})
        assert metrics.macro_f1 == pytest.approx((4 / 5 + 2 / 3) / 2)
        assert metrics.matthews_corrcoef == pytest.approx(2 / math.sqrt(12))
        assert metrics.labels == ["a", "b"]
        assert metrics.confusion_matrix == [[2, 0], [1, 1]]
        assert metrics.sample_count == 4
        assert metrics.mismatch_count == 1
        assert metrics.mismatch_rate == pytest.approx(1 / 4)

    def test_perfect_agreement(self) -> None:
        labels = np.array([0, 0, 1, 2])

        metrics = compute_classification_metrics(labels, labels)

        assert metrics.matthews_corrcoef == pytest.approx(1.0)
        assert metrics.macro_f1 == pytest.approx(1.0)
        assert metrics.mismatch_count == 0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_classification_metrics(np.array([0, 1]), np.array([0]))


class TestBasinFractionComparisons:
    def test_matches_hand_computed_values(self) -> None:
        comparison = compute_basin_fraction_comparisons(
            np.array(["a", "a", "b", "b"]), np.array(["a", "a", "b", "a"])
        )[0]

        actual_e_abs = math.sqrt(0.75 * 0.25 / 4)
        reference_e_abs = math.sqrt(0.5 * 0.5 / 4)
        assert comparison.label == "a"
        assert comparison.actual_bs == pytest.approx(0.75)
        assert comparison.reference_bs == pytest.approx(0.5)
        assert comparison.actual_e_abs == pytest.approx(actual_e_abs)
        assert comparison.reference_e_abs == pytest.approx(reference_e_abs)
        assert comparison.actual_e_rel == pytest.approx(1 / math.sqrt(4 * 0.75))
        assert comparison.reference_e_rel == pytest.approx(1 / math.sqrt(4 * 0.5))
        assert comparison.sampling_uncertainty_z_score == pytest.approx(
            0.25 / math.hypot(actual_e_abs, reference_e_abs)
        )

    def test_class_missing_from_predictions_has_infinite_relative_error(self) -> None:
        comparisons = compute_basin_fraction_comparisons(np.array(["a", "b"]), np.array(["a", "a"]))

        missing = next(c for c in comparisons if c.label == "b")
        assert missing.actual_bs == 0.0
        assert missing.actual_e_rel == math.inf

    def test_identical_labels_score_zero(self) -> None:
        labels = np.array(["a", "a", "a"])

        comparison = compute_basin_fraction_comparisons(labels, labels)[0]

        assert comparison.sampling_uncertainty_z_score == 0.0

    def test_total_disagreement_with_zero_errors_scores_infinite(self) -> None:
        comparisons = compute_basin_fraction_comparisons(
            np.array(["a", "a", "a"]), np.array(["b", "b", "b"])
        )

        assert [c.sampling_uncertainty_z_score for c in comparisons] == [math.inf, math.inf]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_basin_fraction_comparisons(np.array([0, 1]), np.array([0]))


class TestStatisticalComparison:
    @pytest.mark.parametrize(
        ("expected_bs", "expected_z", "passes"),
        [(0.55, 1.0, True), (0.6, 2.0, False)],
    )
    def test_z_score_uses_combined_standard_error(
        self, expected_bs: float, expected_z: float, passes: bool
    ) -> None:
        comparison = compute_statistical_comparison(0.5, 0.03, expected_bs, 0.04)

        assert comparison.z_score == pytest.approx(expected_z)
        assert comparison.passes_test is passes

    def test_difference_with_zero_errors_fails(self) -> None:
        comparison = compute_statistical_comparison(1.0, 0.0, 0.0, 0.0)

        assert comparison.z_score == math.inf
        assert comparison.passes_test is False

    def test_equal_values_with_zero_errors_pass(self) -> None:
        comparison = compute_statistical_comparison(1.0, 0.0, 1.0, 0.0)

        assert comparison.z_score == 0.0
        assert comparison.passes_test is True
