"""Helper utilities for integration tests.

Test Naming Convention:
-----------------------
- test_baseline: Test with default/reference system parameters
- test_parameter_<name>: Test varying a specific system parameter (e.g., test_parameter_T, test_parameter_sigma, test_parameter_v_d)
- test_n<value>: Test with small N for validation (e.g., test_n50, test_n200)
- test_hyperparameter_<name>: Test varying a hyperparameter (e.g., test_hyperparameter_n, test_hyperparameter_rtol)

System Parameter Tests vs Hyperparameter Tests:
------------------------------------------------
System parameter tests vary dynamical system parameters (period T, sigma, velocity v_d, etc.)
and use classification metrics (MCC >= 0.95) for validation.

Hyperparameter tests vary method settings (N, solver tolerance) independent of the
dynamical system. These use the same classification metrics validation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
from test_support import (
    BasinFractionComparison,
    ClassificationMetrics,
    ExecutionTarget,
    assert_predictor_used_backend,
    compute_classification_metrics,
)

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.samplers import CsvSampler
from pybasin.study_params import StudyParams, ZipStudyParams
from pybasin.types import SetupProperties
from pybasin.utils import set_seed


class SetupFunction(Protocol):
    """Case-study setup function with an explicit device override."""

    def __call__(self, device: str | None = None) -> SetupProperties: ...


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
    :param expected_bs: Basin stability loaded from the stored reference JSON.
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


@dataclass
class UnsupervisedBasinFractionComparison(BasinFractionComparison):
    """Comparison metrics for a single attractor in unsupervised clustering.

    Extends BasinFractionComparison with cluster purity information from DBSCAN.

    :ivar dbscan_label: Original DBSCAN cluster label (numeric string).
    :ivar cluster_size: Total trajectories in this cluster.
    :ivar majority_count: Trajectories agreeing with majority template.
    :ivar purity: Fraction agreeing with majority (majority_count / cluster_size).
    """

    dbscan_label: str = ""
    cluster_size: int = 0
    majority_count: int = 0
    purity: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison result for a case study or parameter point.

    :ivar system_name: Name of the dynamical system (e.g., "pendulum", "duffing").
    :ivar case_name: Name of the case (e.g., "case1", "case2").
    :ivar attractors: List of attractor comparisons.
    :ivar parameter_value: Parameter value for parameter sweep tests (None for single-point).
    :ivar classification_metrics: Paired classification evidence, when labels are available.
    :ivar reference_name: Human-readable name for the reference result.
    """

    system_name: str
    case_name: str
    attractors: list[BasinFractionComparison]
    parameter_value: float | None = None
    classification_metrics: ClassificationMetrics | None = None
    reference_name: str = "Stored reference"
    uncertainty_sigma_limit: float = 1.96
    paper_validation: bool = False

    def all_passed(self, mcc_threshold: float = 0.95) -> bool:
        """Check if classification quality is above threshold.

        :param mcc_threshold: MCC threshold for validation.
        :return: True if MCC is above threshold.
        """
        return (
            self.classification_metrics is not None
            and self.classification_metrics.matthews_corrcoef >= mcc_threshold
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "system_name": self.system_name,
            "case_name": self.case_name,
            "parameter_value": self.parameter_value,
            "reference_name": self.reference_name,
            "uncertainty_sigma_limit": self.uncertainty_sigma_limit,
            "classification_metrics": (
                self.classification_metrics.to_dict()
                if self.classification_metrics is not None
                else None
            ),
            "attractors": [a.to_dict() for a in self.attractors],
        }
        if self.paper_validation:
            result["paper_validation"] = True
        return result


@dataclass
class UnsupervisedComparisonResult(ComparisonResult):
    """Comparison result for unsupervised clustering case study.

    Extends ComparisonResult with cluster quality metrics.

    :ivar overall_agreement: Fraction of trajectories where DBSCAN matches KNN.
    :ivar adjusted_rand_index: ARI score comparing DBSCAN to KNN clustering.
    :ivar n_clusters_found: Number of clusters discovered by DBSCAN.
    :ivar n_clusters_expected: Number of clusters expected from reference.
    """

    attractors: list[UnsupervisedBasinFractionComparison] = None  # type: ignore[assignment]
    overall_agreement: float = 0.0
    adjusted_rand_index: float = 0.0
    n_clusters_found: int = 0
    n_clusters_expected: int = 0

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "system_name": self.system_name,
            "case_name": self.case_name,
            "reference_name": self.reference_name,
            "uncertainty_sigma_limit": self.uncertainty_sigma_limit,
            "n_clusters_found": self.n_clusters_found,
            "n_clusters_expected": self.n_clusters_expected,
            "overall_agreement": self.overall_agreement,
            "adjusted_rand_index": self.adjusted_rand_index,
            "classification_metrics": (
                self.classification_metrics.to_dict()
                if self.classification_metrics is not None
                else None
            ),
            "attractors": [a.to_dict() for a in self.attractors],
        }


def run_basin_stability_test(
    json_path: Path,
    setup_function: SetupFunction,
    n: int | None = None,
    seed: int = 42,
    ground_truth_csv: Path | None = None,
    coordinate_columns: list[str] | None = None,
    mcc_threshold: float | None = None,
    execution_target: ExecutionTarget = ExecutionTarget.CPU,
    bs_atol: float | None = None,
    bs_rtol: float = 1.0e-9,
) -> tuple[BasinStabilityEstimator, ComparisonResult]:
    """Run basin stability test with classification metrics validation against ground truth.

    This function:
    1. Loads expected results and metadata (label_map, system_name, case_name) from JSON
    2. Runs every pipeline component on the requested execution target
    3. Verifies N matches between setup and JSON (sum of absNumMembers)
    4. Runs basin stability estimation
    5. Validates results using classification metrics (F1-score, MCC)

    :param json_path: Path to JSON file with expected results.
    :param setup_function: Function that returns system properties.
    :param n: Sample count override. Required for seeded regression tests.
    :param seed: Random seed for deterministic regression tests.
    :param ground_truth_csv: Path to CSV with ground truth labels.
    :param coordinate_columns: Column names to use as coordinates (e.g. ``["x1", "x2"]``).
    :param mcc_threshold: Minimum MCC. Defaults to 0.95.
    :param execution_target: Device on which the solver, sampler, and predictor run.
        Every target reads the same reference coordinates from ``ground_truth_csv`` so
        per-sample classification metrics compare identical initial conditions.
    :param bs_atol: Absolute tolerance for the basin-stability comparison. Defaults to
        the larger of one sample's fraction and 0.001.
    :param bs_rtol: Relative tolerance, same caveat as ``bs_atol``.
    :return: Tuple of (BasinStabilityEstimator, ComparisonResult).
    :raises AssertionError: If validation fails.
    """
    with open(json_path) as f:
        meta = json.load(f)
    label_map: dict[str, str] = meta.get("label_map", {})
    system_name: str = meta.get("system_name", json_path.parent.name)
    case_name: str = meta.get("case_name", "")
    expected_results: list[Any] = meta["data"]

    set_seed(seed)

    # Setup system and run estimation
    props = setup_function(device=execution_target.value)

    solver = props.get("solver")
    assert solver is not None
    solver = solver.clone(device=execution_target.value)

    assert n is not None, "n must be provided for seeded regression tests"
    if ground_truth_csv is None:
        raise ValueError("ground_truth_csv must be provided to load initial conditions")
    if coordinate_columns is None:
        raise ValueError("coordinate_columns must be provided to load initial conditions")
    sampler = CsvSampler(
        ground_truth_csv,
        coordinate_columns=coordinate_columns,
        label_column="label",
        device=execution_target.value,
    )
    expected_n = sum(int(result["absNumMembers"]) for result in expected_results)
    assert expected_n == n, (
        f"Ground truth N mismatch: n={n} but JSON absNumMembers sum={expected_n}"
    )
    assert sampler.n_samples == n, (
        f"Initial-condition CSV contains {sampler.n_samples} rows, expected {n}"
    )
    n_samples = n

    bse = BasinStabilityEstimator(
        n=n_samples,
        ode_system=props["ode_system"],
        sampler=sampler,
        solver=solver,
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        feature_selector=None,
    )

    result = bse.run()
    assert_predictor_used_backend(bse.predictor, "gpu" if execution_target.is_gpu else "cpu")
    basin_stability = result["basin_stability"]

    effective_bs_atol = max(1.0 / n_samples, 0.001)
    if bs_atol is not None:
        effective_bs_atol = bs_atol
    effective_mcc_threshold = 0.95
    if mcc_threshold is not None:
        effective_mcc_threshold = mcc_threshold

    assert bse.y0 is not None
    actual_n = len(bse.y0)
    assert actual_n == expected_n, (
        f"Estimator used {actual_n} initial conditions, expected {expected_n}"
    )
    print(f"\nExpected N: {expected_n}, Actual N: {actual_n}")

    # Get computed standard errors
    errors = result["errors"]

    # Load ground truth labels from CSV
    assert sampler.labels is not None
    y_true = sampler.labels

    # Get predicted labels
    if (
        bse.solution is not None
        and hasattr(bse.solution, "labels")
        and bse.solution.labels is not None
    ):
        y_pred = np.array([str(label) for label in bse.solution.labels])
    else:
        raise ValueError("Estimator solution must have labels after estimation")

    # Compute classification metrics
    metrics = compute_classification_metrics(y_true, y_pred)

    # Build comparison results with F1 per class
    attractor_comparisons: list[BasinFractionComparison] = []

    for expected in expected_results:
        json_label: str = str(expected["label"])
        expected_bs = float(expected["basinStability"])
        expected_std_err = float(expected["standardError"])

        # Skip zero basin stability labels
        if expected_bs == 0:
            continue

        # Map JSON label to Python label
        python_label: str = label_map.get(json_label, json_label)

        # Get actual basin stability for this label
        actual_bs: float = basin_stability.get(python_label, 0.0)
        actual_std_err: float = errors[python_label]["e_abs"] if python_label in errors else 0.0
        actual_relative_error: float = (
            errors[python_label]["e_rel"] if python_label in errors else float("inf")
        )
        expected_relative_error = 1.0 / np.sqrt(n_samples * expected_bs)

        assert_basin_stability_matches(
            actual_bs=actual_bs,
            expected_bs=expected_bs,
            label=python_label,
            context=f"{system_name} {case_name}",
            atol=effective_bs_atol,
            rtol=bs_rtol,
        )

        attractor_comparison = BasinFractionComparison(
            label=python_label,
            actual_bs=actual_bs,
            actual_e_abs=actual_std_err,
            actual_e_rel=actual_relative_error,
            reference_bs=expected_bs,
            reference_e_abs=expected_std_err,
            reference_e_rel=float(expected_relative_error),
        )
        assert attractor_comparison.sampling_uncertainty_z_score <= 1.96, (
            f"Basin stability for {system_name} {case_name}, label={python_label!r}, "
            f"differs from the reference by "
            f"{attractor_comparison.sampling_uncertainty_z_score:.2f} combined standard "
            "errors (limit 1.96)."
        )
        attractor_comparisons.append(attractor_comparison)

    # Verify we have the same labels
    expected_labels = {
        label_map.get(str(result["label"]), str(result["label"]))
        for result in expected_results
        if result["basinStability"] > 0
    }

    actual_labels = {label for label, bs in basin_stability.items() if bs > 0}
    assert expected_labels == actual_labels, (
        f"Label mismatch: expected {expected_labels}, got {actual_labels}"
    )

    # Build comparison result
    comparison_result = ComparisonResult(
        system_name=system_name,
        case_name=case_name,
        attractors=attractor_comparisons,
        parameter_value=None,
        classification_metrics=metrics,
    )

    # Print classification quality summary
    print("\nClassification Metrics:")
    print(f"  Matthews Correlation Coefficient: {metrics.matthews_corrcoef:.4f}")
    for label, f1 in metrics.f1_per_class.items():
        print(f"  F1 ({label}): {f1:.4f}")

    assert metrics.matthews_corrcoef >= effective_mcc_threshold, (
        f"MCC {metrics.matthews_corrcoef:.4f} < {effective_mcc_threshold} "
        f"for {system_name} {case_name}"
    )

    return bse, comparison_result


def run_parameter_study_test(
    json_path: Path,
    setup_function: SetupFunction,
    parameter_name: str,
    n: int | None = None,
    seed: int = 42,
    ground_truths_dir: Path | None = None,
    coordinate_columns: list[str] | None = None,
    mcc_threshold: float | None = None,
    compute_orbit_data: bool = False,
    execution_target: ExecutionTarget = ExecutionTarget.CPU,
    bs_atol: float | None = None,
    bs_rtol: float = 1.0e-9,
) -> tuple[BasinStabilityStudy, list[ComparisonResult]]:
    """Run parameter study test with classification metrics validation against ground truth.

    This function:
    1. Loads expected results and metadata (label_map, system_name, case_name) from JSON
    2. Runs every pipeline component on the requested execution target
    3. Creates and runs BasinStabilityStudy with paired parameters and CSV samplers
    4. For each parameter point, validates results using classification metrics (F1-score, MCC)

    :param json_path: Path to JSON file with expected parameter study results.
    :param setup_function: Function that returns system properties.
    :param parameter_name: Name of parameter to vary.
    :param n: Override sample count per parameter point. Required unless the sweep parameter is n.
    :param seed: Random seed for deterministic regression tests.
    :param ground_truths_dir: Path to directory with parameter_index.csv and param_XXX.csv files.
    :param coordinate_columns: Column names to use as coordinates (e.g. ``["x1", "x2"]``).
    :param mcc_threshold: Minimum MCC. Defaults to 0.95.
    :param compute_orbit_data: Whether to compute orbit data. Pass ``True`` when
        artifact generation is active so orbit diagrams can be plotted.
    :param execution_target: Device on which the solver, sampler, and predictor run.
        Every target reads each parameter point's reference coordinates so per-sample
        classification metrics compare identical initial conditions.
    :param bs_atol: Absolute tolerance for the basin-stability-value comparison against
        the reference JSON.
    :param bs_rtol: Relative tolerance, same caveat as ``bs_atol``.
    :return: Tuple of (BasinStabilityStudy, list of ComparisonResult per parameter).
    :raises AssertionError: If validation fails.
    """
    with open(json_path) as f:
        meta = json.load(f)
    label_map: dict[str, str] = meta.get("label_map", {})
    system_name: str = meta.get("system_name", json_path.parent.name)
    case_name: str = meta.get("case_name", "")
    expected_results: list[Any] = meta["data"]

    set_seed(seed)

    props = setup_function(device=execution_target.value)

    parameter_values_array = np.array([result["parameter"] for result in expected_results])
    label_keys: list[str] = [
        key.replace("bs_", "") for key in expected_results[0] if key.startswith("bs_")
    ]

    solver = props.get("solver")
    feature_extractor = props.get("feature_extractor")
    estimator = props.get("estimator")
    template_integrator = props.get("template_integrator")
    assert solver is not None
    assert feature_extractor is not None
    assert estimator is not None
    solver = solver.clone(device=execution_target.value)

    if ground_truths_dir is None:
        raise ValueError("ground_truths_dir must be provided to compute classification metrics")

    index_file = ground_truths_dir / "parameter_index.csv"
    assert index_file.exists(), f"parameter_index.csv not found in {ground_truths_dir}"
    index_df = pd.read_csv(index_file)  # type: ignore
    gt_csv_paths: list[Path] = []
    for param_val in parameter_values_array:
        closest_idx: int = int(
            np.argmin(
                np.abs(
                    index_df["parameter_value"].values - param_val  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                )
            )
        )
        csv_filename: str = str(index_df.iloc[closest_idx]["filename"])  # type: ignore[reportUnknownMemberType]
        csv_file: Path = ground_truths_dir / csv_filename
        assert csv_file.exists(), f"Ground truth CSV not found: {csv_file}"
        gt_csv_paths.append(csv_file)

    if coordinate_columns is None:
        raise ValueError("coordinate_columns must be provided to load initial conditions")
    device = execution_target.value
    csv_samplers = [
        CsvSampler(p, coordinate_columns=coordinate_columns, label_column="label", device=device)
        for p in gt_csv_paths
    ]
    sampler = csv_samplers[0]
    study_params: StudyParams = ZipStudyParams(**{
        parameter_name: list(parameter_values_array),
        "sampler": csv_samplers,
    })
    effective_n: int = n if n is not None else props["n"]
    for parameter_value, csv_sampler in zip(parameter_values_array, csv_samplers, strict=True):
        expected_sample_count = int(parameter_value) if parameter_name == "n" else effective_n
        assert csv_sampler.n_samples == expected_sample_count, (
            f"Initial-condition CSV for {parameter_name}={parameter_value} contains "
            f"{csv_sampler.n_samples} rows, expected {expected_sample_count}"
        )

    bs_study = BasinStabilityStudy(
        n=effective_n,
        ode_system=props["ode_system"],
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=estimator,
        study_params=study_params,
        template_integrator=template_integrator,
        compute_orbit_data=compute_orbit_data,
    )

    bs_study.run()
    assert_predictor_used_backend(bs_study.estimator, "gpu" if execution_target.is_gpu else "cpu")

    effective_bs_atol = max(1.0 / effective_n, 0.001)
    if bs_atol is not None:
        effective_bs_atol = bs_atol
    effective_mcc_threshold = 0.95
    if mcc_threshold is not None:
        effective_mcc_threshold = mcc_threshold

    # Collect comparison results
    comparison_results: list[ComparisonResult] = []

    # Compare results at each parameter value
    for i, expected in enumerate(expected_results):
        param_value = expected["parameter"]
        actual_bs = bs_study.results[i]["basin_stability"]

        # Get errors for this parameter point
        errors = bs_study.results[i]["errors"]

        # Load ground truth labels and compute metrics
        sampler_labels = csv_samplers[i].labels
        assert sampler_labels is not None
        y_true = sampler_labels

        result_labels_obj = bs_study.results[i]["labels"]
        if result_labels_obj is not None:
            y_pred = np.array([str(label) for label in result_labels_obj])
        else:
            raise ValueError(f"Result {i} must have labels after estimation")

        metrics = compute_classification_metrics(y_true, y_pred)

        # Build attractor comparisons for this parameter point
        attractor_comparisons: list[BasinFractionComparison] = []

        # Check each label
        for label in label_keys:
            bs_key = f"bs_{label}"
            err_key = f"err_{label}"

            expected_bs = float(expected[bs_key])
            expected_err = float(expected.get(err_key, 0.0))

            # Map JSON label to Python label
            python_label: str = label_map.get(label, label)

            assert_basin_stability_matches(
                actual_bs=actual_bs.get(python_label, 0.0),
                expected_bs=expected_bs,
                label=python_label,
                context=(f"{system_name} {case_name} at parameter={float(param_value):.12g}"),
                atol=effective_bs_atol,
                rtol=bs_rtol,
            )

            # Skip zero basin stability labels
            if expected_bs == 0:
                continue

            # Get actual basin stability
            actual_bs_val = actual_bs.get(python_label, 0.0)
            actual_err = errors[python_label]["e_abs"] if python_label in errors else 0.0
            actual_relative_error = (
                errors[python_label]["e_rel"] if python_label in errors else float("inf")
            )
            sample_count = csv_samplers[i].n_samples
            expected_relative_error = 1.0 / np.sqrt(sample_count * expected_bs)

            attractor_comparison = BasinFractionComparison(
                label=python_label,
                actual_bs=actual_bs_val,
                actual_e_abs=actual_err,
                actual_e_rel=actual_relative_error,
                reference_bs=expected_bs,
                reference_e_abs=expected_err,
                reference_e_rel=float(expected_relative_error),
            )
            assert attractor_comparison.sampling_uncertainty_z_score <= 1.96, (
                f"Basin stability for {system_name} {case_name}, label={python_label!r}, "
                f"parameter={float(param_value):.12g}, differs from the reference by "
                f"{attractor_comparison.sampling_uncertainty_z_score:.2f} combined standard "
                "errors (limit 1.96)."
            )
            attractor_comparisons.append(attractor_comparison)

        # Build comparison result for this parameter point
        comparison_results.append(
            ComparisonResult(
                system_name=system_name,
                case_name=case_name,
                attractors=attractor_comparisons,
                parameter_value=param_value,
                classification_metrics=metrics,
            )
        )

    # Print classification quality summary
    print(f"\n{'=' * 80}")
    print("Parameter Study Classification Results")
    print(f"{'=' * 80}")
    mcc_values: list[float] = []
    for _i, result in enumerate(comparison_results):
        param_val = result.parameter_value
        print(f"\nParameter {param_val:.4f}:")
        assert result.classification_metrics is not None
        print(f"  MCC: {result.classification_metrics.matthews_corrcoef:.4f}")
        mcc_values.append(result.classification_metrics.matthews_corrcoef)
    print(f"{'=' * 80}\n")

    # Assert MCC >= threshold for each parameter point (skip mono-stable points where MCC=0.0 is expected)
    for result in comparison_results:
        n_attractors = len(result.attractors)
        if n_attractors >= 2:
            assert result.classification_metrics is not None
            assert result.classification_metrics.matthews_corrcoef >= effective_mcc_threshold, (
                f"MCC {result.classification_metrics.matthews_corrcoef:.4f} "
                f"< {effective_mcc_threshold} "
                f"for {system_name} {case_name} "
                f"at parameter={result.parameter_value}"
            )

    return bs_study, comparison_results


def run_single_point_test(
    n: int,
    expected_bs: dict[str, float],
    setup_function: SetupFunction,
    expected_points: int | None = None,
    seed: int | None = None,
    execution_target: ExecutionTarget = ExecutionTarget.CPU,
) -> None:
    """Run single-point basin stability test with direct value validation.

    This function is for simple tests with one N value and no JSON reference file.
    It directly compares basin stability values without statistical testing.

    :param n: Number of initial conditions to sample.
    :param expected_bs: Expected basin stability values (label -> value).
    :param setup_function: Function that returns system properties.
    :param expected_points: Expected number of points after sampling (for grid samplers).
    :param seed: Optional random seed for reproducibility.
    :param execution_target: Device on which the solver, sampler, and predictor run.
    :raises AssertionError: If validation fails.
    """
    if seed is not None:
        set_seed(seed)

    props = setup_function(device=execution_target.value)

    solver = props.get("solver")
    assert solver is not None
    solver = solver.clone(device=execution_target.value)
    sampler = props["sampler"]

    bse = BasinStabilityEstimator(
        n=n,
        ode_system=props["ode_system"],
        sampler=sampler,
        solver=solver,
        feature_extractor=props.get("feature_extractor"),
        predictor=props.get("estimator"),
        template_integrator=props.get("template_integrator"),
        feature_selector=None,
    )

    basin_stability = bse.run()
    assert_predictor_used_backend(bse.predictor, "gpu" if execution_target.is_gpu else "cpu")

    if bse.y0 is not None:
        actual_points = len(bse.y0)
        print(f"\nActual points generated: {actual_points}")
        if expected_points is not None:
            assert actual_points == expected_points, (
                f"Expected {expected_points} points, but got {actual_points}"
            )

    failures: list[str] = []
    TOLERANCE = 0.05
    for label, expected_value in expected_bs.items():
        actual_value = basin_stability["basin_stability"].get(label, 0.0)
        if abs(actual_value - expected_value) > TOLERANCE:
            failures.append(
                f"Label '{label}': expected {expected_value:.4f}, got {actual_value:.4f}"
            )

    assert not failures, "Basin stability validation failed:\n" + "\n".join(failures)

    total_bs = sum(basin_stability["basin_stability"].values())
    assert abs(total_bs - 1.0) < 0.001, f"Basin stabilities should sum to 1.0, got {total_bs}"
