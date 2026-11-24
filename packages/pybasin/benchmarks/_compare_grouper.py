# pyright: basic
"""Shared comparison/reporting logic for the histogram and pairwise grouper benchmarks.

Both `histogram/compare.py` and `pairwise/compare.py` call `compare_grouper` with their
own grouper name; the result file layout (`julia_<grouper>.json`,
`pybasin_<grouper>_cpu.json`, `pybasin_<grouper>_cuda.json`, all in the same
`{"num_rounds": .., "benchmarks": [{"N": .., "mean_time": .., ...}, ...]}` shape) is
identical between the two, so there's nothing grouper-specific left to duplicate.
"""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from benchmarks.utils import dual_export_figure, load_json

REPO_ROOT: Path = Path(__file__).parent.parent.parent.parent

COLORS: dict[str, str] = {
    "Attractors.jl": "#d95f02",
    "pybasin CPU": "#1b9e77",
    "pybasin CUDA": "#7570b3",
}
IMPL_ORDER: list[str] = ["Attractors.jl", "pybasin CPU", "pybasin CUDA"]


def _extract(results: dict, implementation: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "N": bench["N"],
            "mean_time": bench["mean_time"],
            "std_time": bench["std_time"],
            "implementation": implementation,
        }
        for bench in results["benchmarks"]
    ])


def _build_comparison_figure(df: pd.DataFrame, grouper_name: str) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    n_values: list[int] = sorted(df["N"].unique())
    implementations: list[str] = [
        impl for impl in IMPL_ORDER if impl in df["implementation"].unique()
    ]

    for impl in implementations:
        sub = df[df["implementation"] == impl].set_index("N")
        means = [sub.loc[n, "mean_time"] if n in sub.index else float("nan") for n in n_values]
        ax.plot(n_values, means, "o-", color=COLORS[impl], label=impl)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title(f"{grouper_name.capitalize()} Grouper: pybasin vs Attractors.jl")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    return fig


def compare_grouper(grouper_name: str) -> None:
    """Load, print, and plot the CPU/CUDA/Julia results for one grouper benchmark.

    :param grouper_name: ``"histogram"`` or ``"pairwise"``.
    """
    results_dir = REPO_ROOT / "packages" / "pybasin" / "benchmarks" / grouper_name / "results"
    docs_assets_dir = (
        REPO_ROOT / "packages" / "pybasin" / "docs" / "assets" / "benchmarks" / grouper_name
    )

    sources = {
        "Attractors.jl": results_dir / f"julia_{grouper_name}.json",
        "pybasin CPU": results_dir / f"pybasin_{grouper_name}_cpu.json",
        "pybasin CUDA": results_dir / f"pybasin_{grouper_name}_cuda.json",
    }

    frames: list[pd.DataFrame] = []
    for implementation, path in sources.items():
        if not path.exists():
            print(f"{implementation} results not found at: {path} (skipping)")
            continue
        frames.append(_extract(load_json(path), implementation))

    if not frames:
        print(f"No {grouper_name} benchmark results found in {results_dir}; nothing to compare.")
        return

    df = pd.concat(frames, ignore_index=True)

    print(f"\n=== {grouper_name.capitalize()} Grouper Benchmark Comparison ===")
    print(df.sort_values(["N", "implementation"]).to_string(index=False))

    output_plot = docs_assets_dir / f"{grouper_name}_comparison.png"
    dual_export_figure(
        partial(_build_comparison_figure, df, grouper_name),
        docs_name=output_plot.name,
        docs_dir=output_plot.parent,
        thesis_name=f"{grouper_name}_comparison.pdf",
        thesis_dir=REPO_ROOT / "artifacts" / "benchmarks" / grouper_name,
    )
    print(f"Comparison plot saved to: {output_plot}")

    output_csv = results_dir / f"{grouper_name}_comparison.csv"
    df.to_csv(output_csv, index=False)
    print(f"Comparison data saved to: {output_csv}")
