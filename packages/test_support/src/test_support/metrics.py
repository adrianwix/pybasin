"""Classification and statistical comparison metrics shared by every accuracy suite
(root ``tests/integration`` and each package's own ``tests/accuracy``) that validates
pybasin's output against an external reference implementation.
"""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (  # type: ignore[reportMissingTypeStubs]
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


@dataclass
class StatisticalComparison:
    """Statistical comparison metrics for basin stability values.

    Used when ground truth labels are not available, only expected basin stability values.
    Uses a two-sample z-test with significance level alpha=0.05.

    :ivar z_score: Z-score comparing the actual vs expected basin stability.
    :ivar passes_test: True if |z_score| < 1.96 (alpha=0.05, two-tailed test).
    """

    z_score: float
    passes_test: bool


@dataclass
class ClassificationMetrics:
    """Classification metrics comparing predicted vs ground truth labels.

    :ivar f1_per_class: F1-score for each class label.
    :ivar macro_f1: Unweighted mean of the per-class F1 scores.
    :ivar matthews_corrcoef: Matthews correlation coefficient for overall classification.
    :ivar labels: Label order used by ``f1_per_class`` and ``confusion_matrix``.
    :ivar confusion_matrix: Counts with true labels on rows and predicted labels on columns.
    :ivar sample_count: Number of paired samples compared.
    :ivar mismatch_count: Number of samples assigned different labels.
    :ivar mismatch_rate: Fraction of samples assigned different labels.
    """

    f1_per_class: dict[str, float]
    macro_f1: float
    matthews_corrcoef: float
    labels: list[str]
    confusion_matrix: list[list[int]]
    sample_count: int
    mismatch_count: int
    mismatch_rate: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable classification evidence."""
        return {
            "f1_per_class": self.f1_per_class,
            "macro_f1": self.macro_f1,
            "matthews_corrcoef": self.matthews_corrcoef,
            "labels": self.labels,
            "confusion_matrix": self.confusion_matrix,
            "sample_count": self.sample_count,
            "mismatch_count": self.mismatch_count,
            "mismatch_rate": self.mismatch_rate,
        }


@dataclass
class BasinFractionComparison:
    """Aggregate basin fractions and sampling errors for one paired class."""

    label: str
    actual_bs: float
    actual_e_abs: float
    actual_e_rel: float
    reference_bs: float
    reference_e_abs: float
    reference_e_rel: float

    @property
    def sampling_uncertainty_z_score(self) -> float:
        """Difference in basin fractions measured in combined absolute standard errors."""
        combined_error = float(np.hypot(self.actual_e_abs, self.reference_e_abs))
        difference = abs(self.actual_bs - self.reference_bs)
        return _z_score(difference, combined_error)

    def to_dict(self) -> dict[str, str | float]:
        """Return JSON-serializable basin-fraction evidence."""
        result: dict[str, str | float] = asdict(self)
        result["sampling_uncertainty_z_score"] = self.sampling_uncertainty_z_score
        return result


def _z_score(difference: float, combined_error: float) -> float:
    """Difference in standard errors.

    Both errors are zero when each fraction is exactly 0 or 1, so a nonzero difference
    there is total disagreement and gets an infinite score.
    """
    if combined_error > 0:
        return difference / combined_error
    return float("inf") if difference > 0 else 0.0


def _sampling_errors(basin_fraction: float, sample_count: int) -> tuple[float, float]:
    absolute_error = float(np.sqrt(basin_fraction * (1.0 - basin_fraction) / sample_count))
    relative_error = (
        float(1.0 / np.sqrt(sample_count * basin_fraction))
        if basin_fraction > 0.0
        else float("inf")
    )
    return absolute_error, relative_error


def compute_basin_fraction_comparisons(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[BasinFractionComparison]:
    """Compute basin fractions and formal sampling errors from paired labels."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples."
        )
    sample_count = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))
    comparisons: list[BasinFractionComparison] = []
    for label in labels:
        reference_bs = np.count_nonzero(y_true == label) / sample_count
        actual_bs = np.count_nonzero(y_pred == label) / sample_count
        actual_e_abs, actual_e_rel = _sampling_errors(actual_bs, sample_count)
        reference_e_abs, reference_e_rel = _sampling_errors(reference_bs, sample_count)
        comparisons.append(
            BasinFractionComparison(
                label=str(label),
                actual_bs=actual_bs,
                actual_e_abs=actual_e_abs,
                actual_e_rel=actual_e_rel,
                reference_bs=reference_bs,
                reference_e_abs=reference_e_abs,
                reference_e_rel=reference_e_rel,
            )
        )
    return comparisons


def compute_statistical_comparison(
    actual_bs: float, actual_se: float, expected_bs: float, expected_se: float
) -> StatisticalComparison:
    """Compute statistical comparison metrics for basin stability values.

    Used when ground truth labels are not available, only expected basin stability values.
    Performs a two-sample z-test with significance level alpha=0.05.

    :param actual_bs: Basin stability computed by the implementation under test.
    :param actual_se: Standard error of ``actual_bs``.
    :param expected_bs: Expected basin stability from the reference source.
    :param expected_se: Standard error of the reference basin stability.
    :return: StatisticalComparison with z-score and test result.
    """
    combined_se = float(np.sqrt(actual_se**2 + expected_se**2))
    diff = abs(actual_bs - expected_bs)

    z_score = _z_score(diff, combined_se)

    # Two-tailed z-test at alpha=0.05: critical value is 1.96
    passes_test = z_score < 1.96

    return StatisticalComparison(z_score=z_score, passes_test=passes_test)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ClassificationMetrics:
    """Compute classification metrics between ground truth and predictions.

    Computes per-class and macro F1, MCC, the confusion matrix, and paired
    disagreement counts.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :return: ClassificationMetrics with F1 per class and MCC.
    :raises ValueError: If y_true and y_pred have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples. "
            f"This indicates a bug in label extraction or sample processing."
        )

    true_labels = np.asarray(y_true)
    predicted_labels = np.asarray(y_pred)
    labels = sorted(set(true_labels) | set(predicted_labels))

    f1_per_class_scores = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    if isinstance(f1_per_class_scores, np.ndarray):
        f1_per_class = {
            str(label): float(f1) for label, f1 in zip(labels, f1_per_class_scores, strict=False)
        }
    else:
        f1_per_class = {str(labels[0]): float(f1_per_class_scores)}

    macro_f1 = float(np.mean(list(f1_per_class.values())))
    mcc = float(matthews_corrcoef(true_labels, predicted_labels))
    paired_confusion_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels).astype(
        int
    )
    mismatch_count = int(np.count_nonzero(true_labels != predicted_labels))
    sample_count = len(true_labels)

    return ClassificationMetrics(
        f1_per_class=f1_per_class,
        macro_f1=macro_f1,
        matthews_corrcoef=mcc,
        labels=[str(label) for label in labels],
        confusion_matrix=paired_confusion_matrix.tolist(),
        sample_count=sample_count,
        mismatch_count=mismatch_count,
        mismatch_rate=mismatch_count / sample_count if sample_count else 0.0,
    )


def assert_basin_stability_matches(
    actual_bs: float,
    expected_bs: float,
    label: str,
    context: str,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-9,
) -> None:
    """Assert that a basin stability value matches the stored reference.

    :param actual_bs: Basin stability computed by the current run.
    :param expected_bs: Basin stability loaded from the stored reference.
    :param label: Attractor label being compared.
    :param context: Human-readable context for the assertion message.
    :param atol: Absolute tolerance for floating-point comparison.
    :param rtol: Relative tolerance for floating-point comparison.
    :raises AssertionError: If the two basin stability values differ beyond tolerance.
    """
    assert np.isclose(actual_bs, expected_bs, atol=atol, rtol=rtol), (
        f"Basin stability mismatch for {context}, label='{label}': "
        f"expected {expected_bs:.12f}, got {actual_bs:.12f}"
    )
