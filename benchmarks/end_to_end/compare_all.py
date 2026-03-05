# pyright: basic
"""
Compare benchmark results across all basin stability implementations.

Loads benchmark results from pybasin (Python CPU/CUDA), bSTAB (MATLAB), and optionally
Attractors.jl (Julia) and pynamicalsys. Creates comparison plots and scaling analysis.

Default mode (no flags): pybasin + bSTAB only.
With --all: includes Attractors.jl and pynamicalsys.

Run with:
    uv run python -m benchmarks.end_to_end.compare_all
    uv run python -m benchmarks.end_to_end.compare_all --all
"""

import argparse
from functools import partial
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit

from benchmarks.utils import (
    build_scaling_figure,
    dual_export_figure,
    load_json,
)
from thesis_utils.thesis_plots_utils import (
    THESIS_PALETTE,
    configure_thesis_style,
)

configure_thesis_style()

RESULTS_DIR: Path = Path(__file__).parent / "results"
THESIS_ARTIFACTS_DIR: Path = (
    Path(__file__).parent.parent.parent / "artifacts" / "benchmarks" / "end_to_end"
)
DOCS_ASSETS_DIR: Path = (
    Path(__file__).parent.parent.parent / "docs" / "assets" / "benchmarks" / "end_to_end"
)

COLORS: dict[str, str] = {
    "Python CPU": THESIS_PALETTE[0],
    "Python CUDA": THESIS_PALETTE[2],
    "MATLAB": THESIS_PALETTE[1],
    "Attractors.jl": THESIS_PALETTE[3],
    "pynamicalsys": THESIS_PALETTE[4],
}

MARKERS: dict[str, str] = {
    "Python CPU": "o",
    "Python CUDA": "s",
    "MATLAB": "^",
    "Attractors.jl": "D",
    "pynamicalsys": "v",
}

IMPL_ORDER: list[str] = ["Python CPU", "Python CUDA", "MATLAB", "Attractors.jl", "pynamicalsys"]


def extract_matlab_data(matlab_results: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for bench in matlab_results["benchmarks"]:
        rows.append(
            {
                "N": bench["N"],
                "mean_time": bench["mean_time"],
                "std_time": bench["std_time"],
                "min_time": bench["min_time"],
                "max_time": bench["max_time"],
                "implementation": "MATLAB",
            }
        )
    return pd.DataFrame(rows)


def extract_python_data(python_results: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for bench in python_results["benchmarks"]:
        params = bench["params"]
        stats = bench["stats"]
        rows.append(
            {
                "N": params["n"],
                "mean_time": stats["mean"],
                "std_time": stats["stddev"],
                "min_time": stats["min"],
                "max_time": stats["max"],
                "implementation": f"Python {params['device'].upper()}",
            }
        )
    return pd.DataFrame(rows)


def extract_simple_format_data(results: dict, implementation: str) -> pd.DataFrame:
    """Extract data from MATLAB-style JSON format (used by Julia and pynamicalsys benchmarks)."""
    rows: list[dict] = []
    for bench in results["benchmarks"]:
        rows.append(
            {
                "N": bench["N"],
                "mean_time": bench["mean_time"],
                "std_time": bench["std_time"],
                "min_time": bench["min_time"],
                "max_time": bench["max_time"],
                "implementation": implementation,
            }
        )
    return pd.DataFrame(rows)


def _build_comparison_figure(df: pd.DataFrame) -> Figure:
    """Build the comparison bar chart figure."""
    fig, ax = plt.subplots(figsize=(14, 7))

    n_values: list[int] = sorted(df["N"].unique())
    implementations: list[str] = [
        impl for impl in IMPL_ORDER if impl in df["implementation"].unique()
    ]
    n_impls: int = len(implementations)

    x = np.arange(len(n_values))
    width: float = 0.8 / n_impls

    for i, impl in enumerate(implementations):
        impl_data = df[df["implementation"] == impl].set_index("N")
        means = np.array(
            [impl_data.loc[n, "mean_time"] if n in impl_data.index else 0.0 for n in n_values],
            dtype=float,
        )
        stds = np.array(
            [impl_data.loc[n, "std_time"] if n in impl_data.index else 0.0 for n in n_values],
            dtype=float,
        )
        offset: float = (i - (n_impls - 1) / 2) * width

        ax.bar(
            x + offset,
            means,
            width,
            label=impl,
            color=COLORS[impl],
            yerr=stds,
            capsize=3,
        )

    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("Mean Time (seconds)")
    n_tools: int = len(implementations)
    ax.set_title(f"Basin Stability Computation: {n_tools}-Tool Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in n_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    return fig


def create_comparison_plot(df: pd.DataFrame, output_path: Path) -> None:
    build = partial(_build_comparison_figure, df)
    dual_export_figure(
        build,
        docs_name=output_path.name,
        docs_dir=output_path.parent,
        thesis_name=output_path.stem + ".pdf",
        thesis_dir=THESIS_ARTIFACTS_DIR,
    )
    print(f"Comparison plot saved to: {output_path}")


def print_comparison_table(df: pd.DataFrame) -> None:
    print("\n=== Benchmark Comparison ===")
    df_sorted = df.sort_values(["N", "implementation"])
    print(df_sorted.to_string(index=False))

    print("\n=== Speedup Analysis (vs MATLAB) ===")
    for n in sorted(df["N"].unique()):
        n_data = cast(pd.DataFrame, df[df["N"] == n])
        matlab_rows = cast(pd.DataFrame, n_data[n_data["implementation"] == "MATLAB"])
        if len(matlab_rows) == 0:
            continue
        matlab_time: float = float(matlab_rows["mean_time"].iloc[0])

        others = cast(pd.DataFrame, n_data[n_data["implementation"] != "MATLAB"])
        parts: list[str] = [f"MATLAB={matlab_time:6.2f}s"]
        for _, row in others.iterrows():
            impl_time: float = float(row["mean_time"])
            speedup: float = matlab_time / impl_time if impl_time > 0 else float("inf")
            parts.append(f"{row['implementation']}={impl_time:6.2f}s ({speedup:.2f}x)")

        print(f"N={n:6d}: {', '.join(parts)}")


def analyze_scaling(df: pd.DataFrame) -> dict[str, dict]:
    """
    Analyze time complexity scaling for each implementation.

    Fits T = c * N^alpha (power law) and T = a * N * log(N) + b (linearithmic)
    to determine which model best describes the scaling behavior.
    """
    results: dict[str, dict] = {}
    implementations: list[str] = [
        impl for impl in IMPL_ORDER if impl in df["implementation"].unique()
    ]

    print("\n=== Time Complexity Analysis ===")

    for impl in implementations:
        impl_data = cast(pd.DataFrame, df[df["implementation"] == impl]).sort_values(by="N")
        if len(impl_data) < 3:
            continue

        n_vals = np.asarray(impl_data["N"].values, dtype=float)
        t_vals = np.asarray(impl_data["mean_time"].values, dtype=float)

        log_n = np.log(n_vals)
        log_t = np.log(t_vals)
        result = scipy_stats.linregress(log_n, log_t)
        slope = result.slope
        r_value = result.rvalue
        std_err = result.stderr
        alpha = slope
        dof = len(n_vals) - 2
        t_crit = scipy_stats.t.ppf(0.975, dof)
        alpha_ci = t_crit * std_err
        r2_power = r_value**2

        def linear_model(n: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * n + b

        def nlogn_model(n: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * n * np.log(n) + b

        try:
            popt_linear, _ = curve_fit(linear_model, n_vals, t_vals, p0=[1e-3, 1])
            residuals_linear = t_vals - linear_model(n_vals, *popt_linear)
            ss_res_linear = float(np.sum(residuals_linear**2))
            ss_tot = float(np.sum((t_vals - np.mean(t_vals)) ** 2))
            r2_linear = 1 - ss_res_linear / ss_tot
        except RuntimeError:
            r2_linear = 0.0

        try:
            popt_nlogn, _ = curve_fit(nlogn_model, n_vals, t_vals, p0=[1e-5, 1])
            residuals_nlogn = t_vals - nlogn_model(n_vals, *popt_nlogn)
            ss_res_nlogn = float(np.sum(residuals_nlogn**2))
            ss_tot = float(np.sum((t_vals - np.mean(t_vals)) ** 2))
            r2_nlogn = 1 - ss_res_nlogn / ss_tot
        except RuntimeError:
            r2_nlogn = 0.0

        if alpha < 0.15:
            complexity = "O(1) - constant"
        elif r2_linear > 0.99 and abs(alpha - 1.0) < 0.1:
            complexity = "O(N) - linear"
        elif r2_nlogn > r2_linear and r2_nlogn > 0.99:
            complexity = "O(N log N) - linearithmic"
        elif abs(alpha - 1.0) < 0.15:
            complexity = "O(N) - linear"
        elif abs(alpha - 2.0) < 0.15:
            complexity = "O(N²) - quadratic"
        else:
            complexity = f"O(N^{alpha:.2f})"

        results[impl] = {
            "alpha": alpha,
            "alpha_ci": alpha_ci,
            "r2_power": r2_power,
            "r2_linear": r2_linear,
            "r2_nlogn": r2_nlogn,
            "complexity": complexity,
        }

        print(f"\n{impl}:")
        print(f"  Power law fit: T ∝ N^{alpha:.3f} ± {alpha_ci:.3f} (R² = {r2_power:.4f})")
        print(f"  Linear fit R²: {r2_linear:.4f}")
        print(f"  N log N fit R²: {r2_nlogn:.4f}")
        print(f"  → Scaling: {complexity}")

    return results


def create_scaling_plot(df: pd.DataFrame, output_path: Path) -> None:
    """Create log-log plot showing scaling behavior with fitted lines."""
    build = partial(
        build_scaling_figure,
        df,
        label_column="implementation",
        colors=COLORS,
        markers=MARKERS,
        order=IMPL_ORDER,
        title="Scaling Analysis: Time vs N (log-log)",
    )
    dual_export_figure(
        build,
        docs_name=output_path.name,
        docs_dir=output_path.parent,
        thesis_name=output_path.stem + ".pdf",
        thesis_dir=THESIS_ARTIFACTS_DIR,
    )
    print(f"Scaling plot saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare basin stability benchmark results.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include Attractors.jl and pynamicalsys results (default: pybasin + bSTAB only).",
    )
    args = parser.parse_args()

    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    matlab_json: Path = RESULTS_DIR / "matlab_basin_stability_estimator_scaling.json"
    python_json: Path = RESULTS_DIR / "python_basin_stability_estimator_scaling.json"

    if not matlab_json.exists():
        print(f"MATLAB results not found at: {matlab_json}")
        return
    if not python_json.exists():
        print(f"Python results not found at: {python_json}")
        return

    print(f"Loading MATLAB results from: {matlab_json}")
    matlab_df: pd.DataFrame = extract_matlab_data(load_json(matlab_json))

    print(f"Loading Python results from: {python_json}")
    python_df: pd.DataFrame = extract_python_data(load_json(python_json))

    frames: list[pd.DataFrame] = [matlab_df, python_df]

    if args.all:
        julia_json: Path = RESULTS_DIR / "julia_attractors_basin_stability_estimator_scaling.json"
        pynamicalsys_json: Path = (
            RESULTS_DIR / "pynamicalsys_basin_stability_estimator_scaling.json"
        )

        if julia_json.exists():
            print(f"Loading Attractors.jl results from: {julia_json}")
            frames.append(extract_simple_format_data(load_json(julia_json), "Attractors.jl"))
        else:
            print(f"Attractors.jl results not found at: {julia_json} (skipping)")

        if pynamicalsys_json.exists():
            print(f"Loading pynamicalsys results from: {pynamicalsys_json}")
            frames.append(extract_simple_format_data(load_json(pynamicalsys_json), "pynamicalsys"))
        else:
            print(f"pynamicalsys results not found at: {pynamicalsys_json} (skipping)")

    combined_df: pd.DataFrame = pd.concat(frames, ignore_index=True)

    print_comparison_table(combined_df)
    analyze_scaling(combined_df)

    csv_suffix: str = "_all" if args.all else ""

    output_plot: Path = DOCS_ASSETS_DIR / "end_to_end_comparison.png"
    create_comparison_plot(combined_df, output_plot)

    scaling_plot: Path = DOCS_ASSETS_DIR / "end_to_end_scaling.png"
    create_scaling_plot(combined_df, scaling_plot)

    output_csv: Path = RESULTS_DIR / f"end_to_end_comparison{csv_suffix}.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"Comparison data saved to: {output_csv}")


if __name__ == "__main__":
    main()
