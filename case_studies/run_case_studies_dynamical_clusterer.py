import contextlib
import io
import logging
import os
import warnings
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef  # type: ignore[reportMissingTypeStubs]

from case_studies.duffing_oscillator.main_duffing_dynamical_clusterer import (
    main as duffing_main,
)
from case_studies.friction.main_friction_dynamical_clusterer import main as friction_main
from case_studies.lorenz.main_lorenz_dynamical_clusterer import main as lorenz_main
from case_studies.pendulum.main_pendulum_dynamical_clusterer import main as pendulum_main
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import CsvSampler
from pybasin.types import StudyResult

warnings.filterwarnings("ignore")
os.environ["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS", "cpu")

logging.getLogger("pybasin").setLevel(logging.CRITICAL)
logging.getLogger("jax").setLevel(logging.CRITICAL)
logging.getLogger("diffrax").setLevel(logging.CRITICAL)

TESTS_DIR = Path(__file__).parent.parent / "tests" / "integration"

GT_CSV_PATHS: dict[str, Path] = {
    "DUFFING": TESTS_DIR / "duffing" / "ground_truths" / "main" / "main_duffing.csv",
    "FRICTION": TESTS_DIR / "friction" / "ground_truths" / "main" / "main_friction.csv",
    "PENDULUM": TESTS_DIR / "pendulum" / "ground_truths" / "case1" / "main_pendulum_case1.csv",
    "LORENZ": TESTS_DIR / "lorenz" / "ground_truths" / "main" / "main_lorenz.csv",
}


def _build_label_mapping(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, str]:
    """Map each predicted label to the most frequent ground truth label in that cluster."""
    mapping: dict[str, str] = {}
    for pred_label in np.unique(y_pred):
        mask = y_pred == pred_label
        gt_in_cluster = y_true[mask]
        most_common = str(Counter(gt_in_cluster.tolist()).most_common(1)[0][0])  # pyright: ignore[reportUnknownArgumentType]
        mapping[str(pred_label)] = most_common
    return mapping


def _compute_mcc(
    bse: BasinStabilityEstimator, csv_path: Path, coord_cols: list[str]
) -> float | None:
    if bse.solution is None or bse.solution.labels is None:
        return None
    gt_sampler = CsvSampler(csv_path, coordinate_columns=coord_cols, label_column="label")
    y_true = gt_sampler.labels
    if y_true is None:
        return None
    y_pred = np.array([str(lbl) for lbl in bse.solution.labels])
    mapping = _build_label_mapping(y_pred, y_true)
    y_pred_mapped = np.array([mapping.get(lbl, lbl) for lbl in y_pred])
    return float(matthews_corrcoef(y_true, y_pred_mapped))


def _run_case(
    name: str,
    run_fn: Callable[[], tuple[BasinStabilityEstimator, StudyResult]],
    coord_cols: list[str],
) -> None:
    csv_path = GT_CSV_PATHS[name]
    print(f"\n{'─' * 55}")
    with open(csv_path) as f:
        n_samples = sum(1 for _ in f) - 1
    print(f"  {name}  (n={n_samples})")
    print(f"{'─' * 55}")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        bse, _ = run_fn()
    if bse.result is None:
        print("  No results.")
        return
    for label, val in sorted(
        bse.result["basin_stability"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {label:20s}  {val:.4f}")
    mcc = _compute_mcc(bse, csv_path, coord_cols)
    if mcc is not None:
        print(f"  {'MCC':20s}  {mcc:.4f}")


if __name__ == "__main__":
    print("=" * 55)
    print("  DYNAMICAL CLUSTERER — GROUND TRUTH COMPARISON")
    print("=" * 55)

    _run_case("DUFFING", partial(duffing_main, csv_path=GT_CSV_PATHS["DUFFING"]), ["x1", "x2"])
    _run_case("FRICTION", partial(friction_main, csv_path=GT_CSV_PATHS["FRICTION"]), ["x1", "x2"])
    _run_case("PENDULUM", partial(pendulum_main, csv_path=GT_CSV_PATHS["PENDULUM"]), ["x1", "x2"])
    _run_case("LORENZ", partial(lorenz_main, csv_path=GT_CSV_PATHS["LORENZ"]), ["x1", "x2", "x3"])

    print(f"\n{'=' * 55}")
    print("  DONE")
    print(f"{'=' * 55}")
