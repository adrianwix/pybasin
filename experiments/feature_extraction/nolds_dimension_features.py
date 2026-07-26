# pyright: basic

"""Generate nolds-based nonlinear time-series features from template trajectories.

This script integrates template trajectories provided by the case-study setup
functions (pendulum, lorenz, friction, duffing), extracts a tail segment of
each trajectory, and computes nonlinear features using the ``nolds``
library:

- correlation dimension via ``nolds.corr_dim``
- largest Lyapunov estimates via ``nolds.lyap_r`` and ``nolds.lyap_e``

Features are computed on the first state variable using time-delay embedding,
which reconstructs the full attractor dynamics (Takens' theorem). The features
are assembled into a numeric matrix, missing values are imputed by column medians,
and templates are clustered using ``DBSCAN``. Results (per-template feature values
and cluster assignments) are printed to stdout in a transposed DataFrame format.

Command-line Arguments
----------------------
--emb-dim INT
    Embedding dimension for nolds algorithms (corr_dim, lyap_r, lyap_e).
    Default: 10

--eps FLOAT
    DBSCAN epsilon parameter: maximum distance between two samples to be
    considered neighbors. Smaller values create tighter clusters. Default: 0.5

--min-samples INT
    DBSCAN min_samples parameter: minimum number of samples in a neighborhood
    for a point to be considered a core point. Default: 1

--no-scale
    Skip StandardScaler normalization before clustering. By default, features
    are scaled to zero mean and unit variance.

--verbose
    Enable verbose logging from pybasin modules (solver, template integrator).
    By default, only WARNING level and above are shown.

Requirements: the ``nolds`` package and the project's case-study setup modules.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import logging
import sys
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    import nolds
except ImportError as exc:
    raise ImportError("nolds is required for this experiment (pip install nolds)") from exc

from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from case_studies.friction.setup_friction_system import setup_friction_system
from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system

from pybasin.solution import Solution
from pybasin.template_integrator import TemplateIntegrator
from pybasin.types import SetupProperties

# Suppress nolds warnings about autocorrelation lag and RANSAC consensus
warnings.filterwarnings(
    "ignore", message="autocorrelation declined too slowly", category=RuntimeWarning
)
warnings.filterwarnings("ignore", message="RANSAC did not reach consensus", category=RuntimeWarning)

SetupFactory = Callable[[], "SetupProperties"]


def get_system_setups() -> list[tuple[str, SetupFactory]]:
    return [
        ("pendulum", setup_pendulum_system),
        ("lorenz", setup_lorenz_system),
        ("friction", setup_friction_system),
        ("duffing", setup_duffing_oscillator_system),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate nolds dimension + Lyapunov features for template trajectories before clustering."
    )
    parser.add_argument(
        "--emb-dim", type=int, default=10, help="Embedding dimension for nolds algorithms."
    )
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN eps argument.")
    parser.add_argument("--min-samples", type=int, default=1, help="DBSCAN min_samples argument.")
    parser.add_argument("--no-scale", action="store_true", help="Skip scaler before clustering.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging from pybasin modules."
    )
    return parser.parse_args()


def get_tail_segment(solution: Solution, steady_time: float) -> np.ndarray:
    """Extract trajectory data from steady_time onward."""
    time = solution.time.cpu().numpy()
    y = solution.y.cpu().numpy()
    start_idx = 0
    if steady_time > 0:
        start_idx = np.searchsorted(time, steady_time, side="right")
    return y[start_idx:]


def safe_corr_dim(signal: np.ndarray, emb_dim: int = 10) -> float:
    if np.isnan(signal).any() or np.isinf(signal).any():
        return float("nan")
    if np.std(signal) < 1e-12:
        return 0.0
    try:
        result = nolds.corr_dim(signal, emb_dim=emb_dim)  # type: ignore[no-untyped-call]
        return float(result)  # type: ignore[arg-type]
    except (ValueError, AssertionError):
        return float("nan")


def safe_lyap_r(signal: np.ndarray, emb_dim: int = 10) -> float:
    if np.isnan(signal).any() or np.isinf(signal).any():
        return float("nan")
    if np.std(signal) < 1e-12:
        return 0.0
    try:
        result = nolds.lyap_r(signal, emb_dim=emb_dim, min_tsep=10, trajectory_len=20)  # type: ignore[no-untyped-call]
        return float(result)  # type: ignore[arg-type]
    except (ValueError, AssertionError):
        return float("nan")


def safe_lyap_e(signal: np.ndarray, emb_dim: int = 10) -> float:
    if np.isnan(signal).any() or np.isinf(signal).any():
        return float("nan")
    if np.std(signal) < 1e-12:
        return 0.0
    try:
        result = nolds.lyap_e(signal, emb_dim=emb_dim)  # type: ignore[no-untyped-call]
        return float(result[0])  # type: ignore[arg-type]  # Largest exponent
    except (ValueError, AssertionError):
        return float("nan")


def build_feature_matrix(records: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    if not records:
        return np.empty((0, 0)), []
    feature_names = list(records[0].keys())
    matrix = np.array([[rec[name] for name in feature_names] for rec in records], dtype=float)
    if np.isnan(matrix).any():
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        indices = np.where(np.isnan(matrix))
        matrix[indices] = medians[indices[1]]
    return matrix, feature_names


def classify_attractor_type(label: str) -> str:
    """Determine attractor type from label string."""
    label_lower = label.lower()
    if "fp" in label_lower or "fixed" in label_lower:
        return "fixed_point"
    elif "lc" in label_lower or "period" in label_lower or "limit" in label_lower:
        return "limit_cycle"
    elif "chaos" in label_lower or "chaotic" in label_lower:
        return "chaos"
    else:
        return "unknown"


def validate_feature(feature_name: str, value: float, attractor_type: str) -> str:
    """Check if feature value matches expectations for attractor type."""
    if np.isnan(value):
        return "NaN"

    validators = {
        "corr_dim": {
            "fixed_point": lambda v: "✓" if v < 0.5 else "✗",
            "limit_cycle": lambda v: "✓" if 0.7 < v < 1.5 else "✗",
            "chaos": lambda v: "✓" if v > 1.5 else "✗",
        },
        "lyap_r": {
            "fixed_point": lambda v: "✓" if v < -0.005 else "✗",
            "limit_cycle": lambda v: "✓" if -0.005 <= v <= 0.005 else "✗",
            "chaos": lambda v: "✓" if v > 0.005 else "✗",
        },
    }

    if feature_name in validators and attractor_type in validators[feature_name]:
        return validators[feature_name][attractor_type](value)

    return "-"


def cluster_and_report(
    system_name: str,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    labels: list[str],
    args: argparse.Namespace,
) -> None:
    print(f"\nSystem {system_name}: {len(labels)} templates, features {feature_matrix.shape[1]}")
    if feature_matrix.size == 0:
        print("  No features to cluster")
        return
    if args.no_scale:
        scaled = feature_matrix
    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)
    dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
    cluster_labels = dbscan.fit_predict(scaled)

    # Classify attractor types
    attractor_types = [classify_attractor_type(label) for label in labels]

    # Build DataFrame with labels, clusters, and features
    data = {
        "template": list(range(len(labels))),
        "label": labels,
        "type": attractor_types,
        "cluster": cluster_labels,
    }
    for idx, name in enumerate(feature_names):
        data[name] = feature_matrix[:, idx]
        # Add validation row
        validation_key = f"{name}_valid"
        data[validation_key] = [
            validate_feature(name, feature_matrix[i, idx], attractor_types[i])
            for i in range(len(labels))
        ]

    df = pd.DataFrame(data)

    # Configure pandas display options for better output
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.3f}".format)

    print(df.T.to_string(header=False))

    unique_clusters = sorted({cid for cid in cluster_labels if cid != -1})
    print(f"\nClusters (excluding noise): {unique_clusters}")


def inspect_system(system_name: str, setup_fn: SetupFactory, args: argparse.Namespace) -> None:
    props = setup_fn()
    template_integrator = props.get("template_integrator")
    if not isinstance(template_integrator, TemplateIntegrator):
        print(f"  Skipping {system_name}: no template_integrator available")
        return
    solver = props.get("solver")
    ode_system = props["ode_system"]
    feature_extractor = props.get("feature_extractor")
    if feature_extractor is None:
        raise RuntimeError("feature_extractor is required")
    # Ensure the solver runs on CPU to avoid device mismatches when computing
    # nolds features (these operate on numpy arrays / CPU). Use the solver's
    # `clone()` helper to create a CPU-bound copy if available.
    if solver is None:
        print("  Error: setup did not provide a solver")
        sys.exit(1)

    if hasattr(solver, "clone") and callable(solver.clone):
        solver_cpu = solver.clone(device="cpu")
    else:
        print("  Error: solver has no clone() method - cannot ensure CPU execution. Exiting.")
        sys.exit(1)
    template_integrator.integrate(solver_cpu, ode_system)
    solution = template_integrator.solution
    if solution is None:
        raise RuntimeError("Classifier failed to integrate templates")
    assert feature_extractor.time_steady is not None, "time_steady is not defined"
    tail = get_tail_segment(solution, feature_extractor.time_steady)
    records: list[dict[str, float]] = []
    valid_labels: list[str] = []
    for sample_idx in range(len(template_integrator.labels)):
        sample = tail[:, sample_idx, :]
        # Use first state variable with time-delay embedding to reconstruct attractor
        # (Takens' theorem: embedding reconstructs full dynamics from single observable)
        signal = sample[:, 0]

        # Skip unbounded trajectories (contain inf/nan values)
        if np.isnan(signal).any() or np.isinf(signal).any():
            print(
                f"Template {sample_idx} ({template_integrator.labels[sample_idx]}): SKIPPED (unbounded/divergent)"
            )
            continue

        print(
            f"Template {sample_idx} ({template_integrator.labels[sample_idx]}): signal range = [{signal.min():.3f}, {signal.max():.3f}], mean = {signal.mean():.3f}, std = {signal.std():.3f}"
        )
        record = {
            "corr_dim": safe_corr_dim(signal, emb_dim=args.emb_dim),
            "lyap_r": safe_lyap_r(signal, emb_dim=args.emb_dim),
            "lyap_e": safe_lyap_e(signal, emb_dim=args.emb_dim),
        }
        records.append(record)
        valid_labels.append(template_integrator.labels[sample_idx])

    if not records:
        print(f"  No valid templates for {system_name}")
        return

    matrix, feature_names = build_feature_matrix(records)
    cluster_and_report(system_name, matrix, feature_names, valid_labels, args)


def main() -> None:
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        # Suppress pybasin module logging by default
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("pybasin").setLevel(logging.WARNING)

    for system_name, setup_fn in get_system_setups():
        inspect_system(system_name, setup_fn, args)


if __name__ == "__main__":
    main()
