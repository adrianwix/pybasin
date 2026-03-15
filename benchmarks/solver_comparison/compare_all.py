# pyright: basic
"""
Compare all solver benchmark results: MATLAB, Python, and Julia.

Loads benchmark results from all implementations and creates comparison plots
grouped by N (number of samples). Uses CPSME styling for thesis-quality plots.
"""

from functools import partial
from pathlib import Path
from typing import cast

import pandas as pd

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

THESIS_ARTIFACTS_DIR: Path = (
    Path(__file__).parent.parent.parent / "artifacts" / "benchmarks" / "solver_comparison"
)


def extract_matlab_data(matlab_results: dict) -> pd.DataFrame:
    benchmarks = matlab_results["benchmarks"]

    rows = []
    for bench in benchmarks:
        rows.append(
            {
                "N": bench["n"],
                "mean_time": bench["mean_time"],
                "std_time": bench["std_time"],
                "min_time": bench["min_time"],
                "max_time": bench["max_time"],
                "solver": "MATLAB ode45",
                "device": "cpu",
            }
        )

    return pd.DataFrame(rows)


def extract_python_data(python_results: dict) -> pd.DataFrame:
    benchmarks = python_results["benchmarks"]

    rows = []
    for bench in benchmarks:
        name = bench["name"]
        params = bench["params"]
        stats = bench["stats"]

        if "jax_diffrax" in name:
            solver = "JAX/Diffrax"
        elif "torchdiffeq" in name:
            solver = "torchdiffeq"
        elif "torchode" in name:
            solver = "torchode"
        elif "scipy" in name:
            solver = "scipy"
        else:
            solver = "unknown"

        device = params.get("device", "cuda" if "torchode" in name else "cpu")

        rows.append(
            {
                "N": params["n"],
                "mean_time": stats["mean"],
                "std_time": stats["stddev"],
                "min_time": stats["min"],
                "max_time": stats["max"],
                "solver": solver,
                "device": device,
            }
        )

    return pd.DataFrame(rows)


def extract_scipy_standalone_data(scipy_results: dict) -> pd.DataFrame:
    rows = []
    for bench in scipy_results["benchmarks"]:
        rows.append(
            {
                "N": bench["n"],
                "mean_time": bench["mean"],
                "std_time": bench["std"],
                "min_time": bench["min"],
                "max_time": bench["max"],
                "solver": "scipy",
                "device": "cpu",
            }
        )
    return pd.DataFrame(rows)


LABEL_COLORS: dict[str, str] = {
    "MATLAB ode45 (CPU)": THESIS_PALETTE[1],  # red
    "JAX/Diffrax (CPU)": THESIS_PALETTE[0],  # medium blue
    "JAX/Diffrax (CUDA)": THESIS_PALETTE[6],  # dark blue
    "torchdiffeq (CPU)": THESIS_PALETTE[7],  # medium grey
    "torchdiffeq (CUDA)": THESIS_PALETTE[5],  # teal blue
    "torchode (CUDA)": THESIS_PALETTE[2],  # teal green
    "Julia Ensemble (CPU)": THESIS_PALETTE[3],  # dark grey
    "Julia Ensemble (CUDA)": THESIS_PALETTE[4],  # light blue
    "scipy (CPU)": "#F4A261",  # orange
}

LABEL_MARKERS: dict[str, str] = {
    "MATLAB ode45 (CPU)": "^",
    "JAX/Diffrax (CPU)": "o",
    "JAX/Diffrax (CUDA)": "s",
    "torchdiffeq (CPU)": "v",
    "torchdiffeq (CUDA)": "D",
    "torchode (CUDA)": "P",
    "Julia Ensemble (CPU)": "X",
    "Julia Ensemble (CUDA)": "*",
    "scipy (CPU)": "h",
}

LABEL_ORDER: list[str] = [
    "MATLAB ode45 (CPU)",
    "JAX/Diffrax (CPU)",
    "JAX/Diffrax (CUDA)",
    "torchdiffeq (CPU)",
    "torchdiffeq (CUDA)",
    "torchode (CUDA)",
    "Julia Ensemble (CPU)",
    "Julia Ensemble (CUDA)",
    "scipy (CPU)",
]


def _solver_label(solver: str, device: str) -> str:
    return f"{solver} ({device.upper()})"


def extract_julia_ensemble_data(julia_results: dict, device: str = "cpu") -> pd.DataFrame:
    benchmarks = julia_results["benchmarks"]

    rows = []
    for bench in benchmarks:
        rows.append(
            {
                "N": bench["N"],
                "mean_time": bench["mean_time"],
                "std_time": bench["std_time"],
                "min_time": bench["min_time"],
                "max_time": bench["max_time"],
                "solver": "Julia Ensemble",
                "device": device,
            }
        )

    return pd.DataFrame(rows)


def _add_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with a ``label`` column derived from solver + device."""
    df = df.copy()
    df["label"] = df.apply(lambda r: _solver_label(r["solver"], r["device"]), axis=1)
    return df


def create_scaling_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create log-log scaling plot with regression lines."""
    labelled = _add_label_column(df)
    build = partial(
        build_scaling_figure,
        labelled,
        label_column="label",
        colors=LABEL_COLORS,
        markers=LABEL_MARKERS,
        order=LABEL_ORDER,
        title="ODE Solver Scaling: Time vs N (log-log)",
    )
    dual_export_figure(
        build,
        docs_name="solver_scaling.png",
        docs_dir=output_dir,
        thesis_name="solver_scaling.pdf",
        thesis_dir=THESIS_ARTIFACTS_DIR,
    )
    print(f"Scaling plot saved to: {output_dir / 'solver_scaling.png'}")


def print_comparison_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SOLVER BENCHMARK COMPARISON")
    print("=" * 80)

    for n in sorted(df["N"].unique()):
        print(f"\n--- N = {n:,} ---")
        n_data = cast(pd.DataFrame, df[df["N"] == n]).sort_values(by="mean_time")

        fastest_time = float(n_data["mean_time"].min())

        for _, row in n_data.iterrows():
            speedup = row["mean_time"] / fastest_time
            speedup_str = f"({speedup:.1f}x slower)" if speedup > 1.01 else "(fastest)"
            print(
                f"  {row['solver']:15} ({row['device']:4}): "
                f"{row['mean_time']:8.2f} ± {row['std_time']:.2f}s  {speedup_str}"
            )

    print("\n" + "=" * 80)
    print("SPEEDUP vs MATLAB ode45")
    print("=" * 80)

    for n in sorted(df["N"].unique()):
        n_data = cast(pd.DataFrame, df[df["N"] == n])
        matlab_data = cast(pd.DataFrame, n_data[n_data["solver"] == "MATLAB ode45"])

        if len(matlab_data) == 0:
            continue

        matlab_time = float(matlab_data["mean_time"].iloc[0])
        print(f"\n--- N = {n:,} (MATLAB baseline: {matlab_time:.2f}s) ---")

        python_data = cast(pd.DataFrame, n_data[n_data["solver"] != "MATLAB ode45"]).sort_values(
            by="mean_time"
        )
        for _, row in python_data.iterrows():
            speedup = matlab_time / row["mean_time"]
            direction = "faster" if speedup > 1 else "slower"
            print(
                f"  {row['solver']:15} ({row['device']:4}): "
                f"{row['mean_time']:8.2f}s  → {abs(speedup):.2f}x {direction}"
            )


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    docs_assets_dir = (
        Path(__file__).parent.parent.parent / "docs" / "assets" / "benchmarks" / "solver_comparison"
    )
    docs_assets_dir.mkdir(parents=True, exist_ok=True)
    THESIS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    matlab_json = results_dir / "matlab_benchmark_results.json"
    python_json = results_dir / "python_benchmark_results.json"
    scipy_json = results_dir / "python_scipy_benchmark_results.json"
    julia_ensemble_json = results_dir / "julia_ensemble_basin_stability_scaling.json"
    julia_ensemble_gpu_json = results_dir / "julia_ensemble_gpu_basin_stability_scaling.json"

    frames: list[pd.DataFrame] = []

    if matlab_json.exists():
        print(f"Loading MATLAB results from: {matlab_json}")
        frames.append(extract_matlab_data(load_json(matlab_json)))
    else:
        print(f"MATLAB results not found at: {matlab_json} (skipping)")

    if python_json.exists():
        print(f"Loading Python results from: {python_json}")
        frames.append(extract_python_data(load_json(python_json)))
    else:
        print(f"Python results not found at: {python_json} (skipping)")

    if scipy_json.exists():
        print(f"Loading scipy standalone results from: {scipy_json}")
        frames.append(extract_scipy_standalone_data(load_json(scipy_json)))
    else:
        print(f"scipy standalone results not found at: {scipy_json} (skipping)")

    if julia_ensemble_json.exists():
        print(f"Loading Julia Ensemble (CPU) results from: {julia_ensemble_json}")
        frames.append(extract_julia_ensemble_data(load_json(julia_ensemble_json), device="cpu"))
    else:
        print(f"Julia Ensemble (CPU) results not found at: {julia_ensemble_json} (skipping)")

    if julia_ensemble_gpu_json.exists():
        print(f"Loading Julia Ensemble (GPU) results from: {julia_ensemble_gpu_json}")
        frames.append(
            extract_julia_ensemble_data(load_json(julia_ensemble_gpu_json), device="cuda")
        )
    else:
        print(f"Julia Ensemble (GPU) results not found at: {julia_ensemble_gpu_json} (skipping)")

    if not frames:
        print("No benchmark results found. Run at least one benchmark first.")
        return

    combined_df = pd.concat(frames, ignore_index=True)

    print_comparison_table(combined_df)

    create_scaling_plot(combined_df, docs_assets_dir)

    output_csv = results_dir / "solver_comparison.csv"
    combined_df.to_csv(output_csv, index=False)
    print(f"\nComparison data saved to: {output_csv}")


if __name__ == "__main__":
    main()
