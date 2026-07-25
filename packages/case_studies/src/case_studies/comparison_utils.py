import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from pybasin.types import ErrorInfo


class ExpectedItem(TypedDict):
    label: str
    basinStability: float
    standardError: float


def compare_with_expected_by_size(
    basin_stability: dict[str, float],
    expected_file_path: Path,
    errors: Mapping[str, ErrorInfo] | None = None,
) -> None:
    """Compare basin stability results with expected values by matching cluster sizes.

    Sorts both actual and expected results by size (descending) and compares them
    in order, avoiding issues with arbitrary cluster ID ordering.

    Args:
        basin_stability: Actual basin stability values.
        expected_file_path: Path to JSON file with expected results.
        errors: Optional error dict from bse.get_errors() with 'e_abs' per label.
    """
    if not expected_file_path.exists():
        print(f"Expected file not found: {expected_file_path}")
        return

    with open(expected_file_path) as f:
        expected_data: list[dict[str, Any]] = json.load(f)

    actual_sorted = sorted(basin_stability.items(), key=lambda x: x[1], reverse=True)
    expected_filtered: list[ExpectedItem] = [
        {
            "label": item["label"],
            "basinStability": item["basinStability"],
            "standardError": item.get("standardError", 0.0),
        }
        for item in expected_data
        if item["label"] != "NaN"
    ]
    expected_sorted = sorted(expected_filtered, key=lambda x: x["basinStability"], reverse=True)

    if errors is not None:
        default_error: ErrorInfo = {"e_abs": 0.0, "e_rel": 0.0}
        errors_sorted: list[ErrorInfo] = [
            errors.get(label, default_error) for label, _ in actual_sorted
        ]
        _print_comparison_with_errors(actual_sorted, expected_sorted, errors_sorted, by_size=True)
    else:
        _print_comparison_simple(actual_sorted, expected_sorted, by_size=True)


def compare_with_expected(
    basin_stability: dict[str, float],
    label_mapping: dict[str, str],
    expected_file_path: Path,
    errors: Mapping[str, ErrorInfo] | None = None,
) -> None:
    """Compare basin stability results with expected values using label mapping.

    Args:
        basin_stability: Actual basin stability values.
        label_mapping: Mapping from actual labels to expected labels.
        expected_file_path: Path to JSON file with expected results.
        errors: Optional error dict from bse.get_errors() with 'e_abs' per label.
    """
    mapped_results = {label_mapping.get(k, k): float(v) for k, v in basin_stability.items()}
    mapped_errors: dict[str, ErrorInfo] | None = (
        {label_mapping.get(k, k): v for k, v in errors.items()} if errors else None
    )

    if not expected_file_path.exists():
        print(f"Expected file not found: {expected_file_path}")
        return

    with open(expected_file_path) as f:
        expected_data: list[dict[str, Any]] = json.load(f)

    expected_results: dict[str, ExpectedItem] = {}
    for item in expected_data:
        if item["label"] != "NaN":
            expected_results[item["label"]] = ExpectedItem(
                label=item["label"],
                basinStability=item["basinStability"],
                standardError=item.get("standardError", 0.0),
            )

    labels_to_compare = sorted([label for label in set(label_mapping.values()) if label != "NaN"])

    actual_sorted: list[tuple[str, float]] = [
        (label, mapped_results.get(label, 0.0)) for label in labels_to_compare
    ]

    default_expected: ExpectedItem = {"label": "", "basinStability": 0.0, "standardError": 0.0}
    expected_sorted: list[ExpectedItem] = [
        ExpectedItem(
            label=label,
            basinStability=expected_results.get(label, default_expected)["basinStability"],
            standardError=expected_results.get(label, default_expected)["standardError"],
        )
        for label in labels_to_compare
    ]

    if mapped_errors is not None:
        default_error: ErrorInfo = {"e_abs": 0.0, "e_rel": 0.0}
        errors_sorted: list[ErrorInfo] = [
            mapped_errors.get(label, default_error) for label in labels_to_compare
        ]
        _print_comparison_with_errors(actual_sorted, expected_sorted, errors_sorted, by_size=False)
    else:
        _print_comparison_simple(actual_sorted, expected_sorted, by_size=False)


def _print_comparison_simple(
    actual_sorted: list[tuple[str, float]],
    expected_sorted: list[ExpectedItem],
    by_size: bool,
) -> None:
    """Print simple comparison without error analysis."""
    title = "COMPARISON WITH EXPECTED RESULTS" + (" (sorted by size)" if by_size else "")
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for (cluster_id, actual), expected_item in zip(actual_sorted, expected_sorted, strict=False):
        expected = expected_item["basinStability"]
        label = expected_item["label"]
        diff = actual - expected
        if by_size:
            print(
                f"Cluster {cluster_id} -> {label:15s}: "
                f"Actual={actual:.1%}  Expected={expected:.1%}  Diff={diff:+.1%}"
            )
        else:
            print(f"{label:20s}: Actual={actual:.1%}  Expected={expected:.1%}  Diff={diff:+.1%}")

    if len(actual_sorted) != len(expected_sorted):
        print(f"Warning: {len(actual_sorted)} clusters found, {len(expected_sorted)} expected")

    print("=" * 60 + "\n")


def _print_comparison_with_errors(
    actual_sorted: list[tuple[str, float]],
    expected_sorted: list[ExpectedItem],
    errors_sorted: list[ErrorInfo],
    by_size: bool,
) -> None:
    """Print comparison with error analysis."""
    title = "COMPARISON WITH EXPECTED RESULTS" + (" (sorted by size)" if by_size else "")
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"{'Label':<20}  {'Actual':>8}  {'Expected':>8}  {'Diff':>8}  "
        f"{'Comb.SE':>8}  {'Sigma':>6}  {'Status':<10}"
    )
    print("-" * 78)

    all_within_bounds = True
    for (cluster_id, actual), expected_item, error_info in zip(
        actual_sorted, expected_sorted, errors_sorted, strict=False
    ):
        expected = expected_item["basinStability"]
        label = expected_item["label"]
        expected_se = expected_item.get("standardError", 0.0)
        actual_se = error_info.get("e_abs", 0.0)

        diff = actual - expected
        combined_se = math.sqrt(actual_se**2 + expected_se**2)

        if combined_se > 0:  # noqa: SIM108
            sigma = abs(diff) / combined_se
        else:
            sigma = 0.0 if diff == 0 else float("inf")

        if sigma <= 2:
            status = "OK"
        elif sigma <= 3:
            status = f"~{sigma:.1f}s"
            all_within_bounds = False
        else:
            status = f"{sigma:.1f}s !"
            all_within_bounds = False

        display_label = f"{cluster_id} -> {label}" if by_size else label
        print(
            f"{display_label:<20}  {actual:>7.1%}   {expected:>7.1%}   {diff:>+7.1%}   "
            f"{combined_se:>7.2%}   {sigma:>5.1f}   {status:<10}"
        )

    if len(actual_sorted) != len(expected_sorted):
        print(f"Warning: {len(actual_sorted)} clusters found, {len(expected_sorted)} expected")

    print("-" * 78)
    if all_within_bounds:
        print("All differences are within 2 sigma (statistically consistent)")
    else:
        print("Some differences exceed 2 sigma (may indicate systematic differences)")
    print("=" * 78 + "\n")
