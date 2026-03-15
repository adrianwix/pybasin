# pyright: basic
"""MkDocs macros for case study documentation.

This module provides macros for rendering comparison tables from JSON artifacts
and loading code snippets from source files.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "results"


def _format_bs_with_se(bs: float, se: float) -> str:
    """Format basin stability with standard error."""
    return f"{bs:.5f} ± {se:.5f}"


def comparison_table(case_id: str) -> str:
    """Render a comparison table from a JSON artifact.

    For single-point tests, renders a table with columns:
    Attractor | pybasin BS ± SE | bSTAB BS ± SE

    For parameter sweep tests, adds a Parameter column first:
    Parameter | Attractor | pybasin BS ± SE | bSTAB BS ± SE

    For unsupervised tests, adds cluster quality metrics and purity column:
    Attractor | DBSCAN | Purity | pybasin BS ± SE | bSTAB BS ± SE

    For paper validation tests (no ground truth labels), shows confidence intervals:
    Attractor | pybasin BS ± SE | Paper BS ± SE | Difference | 95% CI | Status

    Also shows overall MCC in a summary section.

    :param case_id: Case identifier (e.g., "pendulum_case1", "pendulum_case2").
    :return: Markdown table string.
    """
    json_path = ARTIFACTS_DIR / f"{case_id}_comparison.json"

    if not json_path.exists():
        return f'!!! warning "Missing Data"\n    Comparison data not found: `{case_id}_comparison.json`\n    Run tests with `--generate-artifacts` to generate.'

    with open(json_path) as f:
        data: dict[str, Any] = json.load(f)

    # Detect paper validation case: explicit flag in JSON
    is_paper_validation = data.get("paper_validation", False)
    if not is_paper_validation and "parameter_results" in data:
        param_results: list[dict[str, Any]] = data["parameter_results"]
        if param_results:
            is_paper_validation = param_results[0].get("paper_validation", False)

    if is_paper_validation:
        if "parameter_results" in data:
            return _render_paper_validation_sweep_table(data)
        return _render_paper_validation_table(data)

    if "parameter_results" in data:
        return _render_parameter_sweep_table(data)
    if "overall_agreement" in data:
        return _render_unsupervised_table(data)
    return _render_single_point_table(data)


def _render_paper_validation_table(data: dict[str, Any]) -> str:
    """Render table for paper validation (statistical comparison)."""
    attractors: list[dict[str, Any]] = data.get("attractors", [])

    if not attractors:
        return '!!! warning "No Data"\n    No attractor data found in comparison.'

    # Table header
    table_lines: list[str] = [
        "| Attractor | pybasin BS ± SE | Paper BS ± SE | Difference | 95% CI | Status |",
        "|-----------|-----------------|---------------|------------|--------|--------|",
    ]

    for a in attractors:
        python_bs: float = a["python_bs"]
        python_se: float = a["python_se"]
        matlab_bs: float = a["matlab_bs"]
        matlab_se: float = a["matlab_se"]

        python_str = _format_bs_with_se(python_bs, python_se)
        matlab_str = _format_bs_with_se(matlab_bs, matlab_se)

        # Compute difference and 95% confidence interval
        diff = python_bs - matlab_bs
        # For 95% confidence, z = 1.96
        combined_se = (python_se**2 + matlab_se**2) ** 0.5
        ci_margin = 1.96 * combined_se

        # Check if difference is within CI (difference should contain 0)
        within_ci = abs(diff) <= ci_margin
        status = "✓" if within_ci else "✗"

        diff_str = f"{diff:+.4f}"
        ci_str = f"±{ci_margin:.4f}"

        table_lines.append(
            f"| {a['label']} | {python_str} | {matlab_str} | {diff_str} | {ci_str} | {status} |"
        )

    return "\n".join(table_lines)


def _render_single_point_table(data: dict[str, Any]) -> str:
    """Render table for single-point comparison."""
    attractors: list[dict[str, Any]] = data.get("attractors", [])

    if not attractors:
        return '!!! warning "No Data"\n    No attractor data found in comparison.'

    # Get overall metrics
    mcc = data.get("matthews_corrcoef", 0.0)

    # Summary metrics
    summary_lines: list[str] = [
        "**Overall Classification Quality:**",
        "",
        f"- Matthews Correlation Coefficient: {mcc:.4f}",
        "",
    ]

    # Attractor table
    table_lines: list[str] = [
        "| Attractor | pybasin BS ± SE | bSTAB BS ± SE |",
        "|-----------|-----------------|---------------|",
    ]

    for a in attractors:
        python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
        matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])

        table_lines.append(f"| {a['label']} | {python_str} | {matlab_str} |")

    return "\n".join(summary_lines + table_lines)


def _render_unsupervised_table(data: dict[str, Any]) -> str:
    """Render table for unsupervised clustering comparison."""
    attractors: list[dict[str, Any]] = data.get("attractors", [])

    if not attractors:
        return '!!! warning "No Data"\n    No attractor data found in comparison.'

    # Cluster quality metrics summary
    n_found = data.get("n_clusters_found", 0)
    n_expected = data.get("n_clusters_expected", 0)
    agreement = data.get("overall_agreement", 0.0)
    ari = data.get("adjusted_rand_index", 0.0)
    mcc = data.get("matthews_corrcoef", 0.0)

    summary_lines: list[str] = [
        "**Cluster Quality Metrics:**",
        "",
        f"- Clusters found: {n_found} (expected: {n_expected})",
        f"- Overall agreement: {agreement:.1%}",
        f"- Adjusted Rand Index: {ari:.4f}",
        f"- Matthews Correlation Coefficient: {mcc:.4f}",
        "",
    ]

    # Attractor table with purity info
    table_lines: list[str] = [
        "| Attractor | DBSCAN | Purity | pybasin BS ± SE | bSTAB BS ± SE |",
        "|-----------|--------|--------|-----------------|---------------|",
    ]

    for a in attractors:
        python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
        matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])
        dbscan_label = a.get("dbscan_label", "-")
        purity = a.get("purity", 0.0)
        purity_str = f"{purity:.1%}"

        table_lines.append(
            f"| {a['label']} | {dbscan_label} | {purity_str} | {python_str} | {matlab_str} |"
        )

    return "\n".join(summary_lines + table_lines)


def _render_paper_validation_sweep_table(data: dict[str, Any]) -> str:
    """Render table for paper validation parameter sweep as a single consolidated table."""
    parameter_results: list[dict[str, Any]] = data.get("parameter_results", [])

    if not parameter_results:
        return '!!! warning "No Data"\n    No parameter data found in comparison.'

    param_name: str = data.get("parameter_name", "Parameter")

    # Build single consolidated table
    table_lines: list[str] = [
        f"| {param_name} | Attractor | pybasin BS ± SE | Paper BS ± SE | Difference | 95% CI | Status |",
        "|-------------|-----------|-----------------|---------------|------------|--------|--------|",
    ]

    for result in parameter_results:
        param_value = result.get("parameter_value")
        if param_value is None:
            param_str = "-"
        elif isinstance(param_value, float):
            if param_value < 0.01:
                param_str = f"{param_value:.1e}"
            else:
                param_str = f"{param_value:.4f}".rstrip("0").rstrip(".")
        else:
            param_str = str(param_value)

        attractors: list[dict[str, Any]] = result.get("attractors", [])
        for i, a in enumerate(attractors):
            python_bs: float = a["python_bs"]
            python_se: float = a["python_se"]
            matlab_bs: float = a["matlab_bs"]
            matlab_se: float = a["matlab_se"]

            python_str = _format_bs_with_se(python_bs, python_se)
            matlab_str = _format_bs_with_se(matlab_bs, matlab_se)

            # Compute difference and 95% confidence interval
            diff = python_bs - matlab_bs
            combined_se = (python_se**2 + matlab_se**2) ** 0.5
            ci_margin = 1.96 * combined_se

            within_ci = abs(diff) <= ci_margin
            status = "✓" if within_ci else "✗"

            diff_str = f"{diff:+.5f}"
            ci_str = f"±{ci_margin:.5f}"

            if i == 0:
                table_lines.append(
                    f"| {param_str} | {a['label']} | {python_str} | {matlab_str} | {diff_str} | {ci_str} | {status} |"
                )
            else:
                table_lines.append(
                    f"| | {a['label']} | {python_str} | {matlab_str} | {diff_str} | {ci_str} | |"
                )

    return "\n".join(table_lines)


def _render_parameter_sweep_table(data: dict[str, Any]) -> str:
    """Render table for parameter sweep comparison as a single consolidated table."""
    parameter_results: list[dict[str, Any]] = data.get("parameter_results", [])

    if not parameter_results:
        return '!!! warning "No Data"\n    No parameter data found in comparison.'

    param_name: str = data.get("parameter_name", "Parameter")

    # Compute average MCC, excluding trivial cases (single attractor with matching BS)
    mcc_values_for_avg: list[float] = []
    excluded_count = 0

    for result in parameter_results:
        mcc = result.get("matthews_corrcoef", 0.0)
        attractors = result.get("attractors", [])

        # Check if this is a trivial case: single attractor with matching BS
        is_trivial = False
        if len(attractors) == 1:
            a = attractors[0]
            if abs(a["python_bs"] - a["matlab_bs"]) < 1e-8:
                is_trivial = True
                excluded_count += 1

        if not is_trivial:
            mcc_values_for_avg.append(mcc)

    # Build header with average
    header_lines: list[str] = []
    if mcc_values_for_avg:
        avg_mcc = sum(mcc_values_for_avg) / len(mcc_values_for_avg)
        header_lines.append(f"**Average MCC = {avg_mcc:.4f}**")
        if excluded_count > 0:
            header_lines.append("")
            header_lines.append(
                "*The average excludes cases where there is only a single attractor and the basin stability "
                "values are the same since MCC is 0 for single class cases, and would therefore drop the average.*"
            )
        header_lines.append("")

    # Build single consolidated table
    table_lines: list[str] = [
        f"| {param_name} | Attractor | pybasin BS ± SE | bSTAB BS ± SE | MCC |",
        "|-------------|-----------|-----------------|---------------|-----|",
    ]

    for result in parameter_results:
        param_value = result.get("parameter_value")
        if param_value is None:
            param_str = "-"
        elif isinstance(param_value, float):
            if param_value < 0.01:
                param_str = f"{param_value:.1e}"
            else:
                param_str = f"{param_value:.4f}".rstrip("0").rstrip(".")
        else:
            param_str = str(param_value)

        mcc = result.get("matthews_corrcoef", 0.0)
        attractors: list[dict[str, Any]] = result.get("attractors", [])

        for i, a in enumerate(attractors):
            python_str = _format_bs_with_se(a["python_bs"], a["python_se"])
            matlab_str = _format_bs_with_se(a["matlab_bs"], a["matlab_se"])

            if i == 0:
                table_lines.append(
                    f"| {param_str} | {a['label']} | {python_str} | {matlab_str} | {mcc:.4f} |"
                )
            else:
                table_lines.append(f"| | {a['label']} | {python_str} | {matlab_str} | |")

    return "\n".join(header_lines + table_lines)


def load_snippet(spec: str) -> str:
    """Load a code snippet from a source file.

    :param spec: Specification in format "path/to/file.py::function_name"
                 Path should be relative to the workspace root.
    :return: Markdown-formatted code block with the extracted function.
    """
    try:
        file_path_str, func_name = spec.split("::")
    except ValueError:
        return f'!!! error "Invalid Format"\n    Expected format: `path/to/file.py::function_name`\n    Got: `{spec}`'

    workspace_root = Path(__file__).parent.parent
    file_path = workspace_root / file_path_str

    if not file_path.exists():
        return f'!!! error "File Not Found"\n    Could not find file: `{file_path_str}`'

    try:
        source_code = file_path.read_text()
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                lines = source_code.splitlines()
                start_line = node.lineno - 1
                end_line = node.end_lineno if node.end_lineno else len(lines)

                function_code = "\n".join(lines[start_line:end_line])

                return f"```python\n{function_code}\n```"

        return f'!!! warning "Function Not Found"\n    Could not find function `{func_name}` in `{file_path_str}`'

    except SyntaxError as e:
        return f'!!! error "Syntax Error"\n    Failed to parse `{file_path_str}`: {e}'
    except Exception as e:
        return f'!!! error "Error"\n    Failed to load snippet: {e}'


BENCHMARK_RESULTS_DIR = Path(__file__).parent.parent / "benchmarks" / "end_to_end" / "results"
SOLVER_COMPARISON_RESULTS_DIR = (
    Path(__file__).parent.parent / "benchmarks" / "solver_comparison" / "results"
)


SOLVER_IMPLEMENTATIONS: list[tuple[str, str, str]] = [
    ("MATLAB ode45", "cpu", "MATLAB ode45"),
    ("JAX/Diffrax", "cpu", "JAX (CPU)"),
    ("JAX/Diffrax", "cuda", "JAX (CUDA)"),
    ("torchdiffeq", "cpu", "torchdiffeq (CPU)"),
    ("torchdiffeq", "cuda", "torchdiffeq (CUDA)"),
    ("torchode", "cuda", "torchode (CUDA)"),
    ("Julia Ensemble", "cpu", "Julia (CPU)"),
    ("Julia Ensemble", "cuda", "Julia (GPU)"),
    ("scipy", "cpu", "scipy (CPU)"),
]


def _fmt_solver_cell(
    t: float, baseline_t: float | None, rank: int, is_baseline: bool = False
) -> str:
    """Format a solver table cell with optional speedup and rank styling.

    :param t: Time value in seconds.
    :param baseline_t: Baseline time for speedup calculation.
    :param rank: 1=bold (fastest), 2=italic (second fastest), 0=normal.
    :param is_baseline: If True, don't show speedup.
    :return: Formatted Markdown cell string.
    """
    if is_baseline:
        cell = f"{t:.3f}s"
    elif baseline_t is not None:
        cell = f"{t:.3f}s ({baseline_t / t:.3f}×)"
    else:
        cell = f"{t:.3f}s"
    if rank == 1:
        return f"**{cell}**"
    elif rank == 2:
        return f"*{cell}*"
    return cell


def solver_comparison_table() -> str:
    """Render solver comparison table from CSV data.

    Solvers as rows, N values as columns. Each non-baseline cell shows time and
    speedup relative to MATLAB ode45. Bold marks the fastest solver per column,
    italic marks the second fastest.

    :return: Markdown table string.
    """
    csv_path = SOLVER_COMPARISON_RESULTS_DIR / "solver_comparison.csv"

    if not csv_path.exists():
        return '!!! warning "Missing Data"\n    Solver comparison data not found. Run `uv run python benchmarks/solver_comparison/compare_all.py` to generate.'

    df = pd.read_csv(csv_path)

    solver_keys: list[tuple[str, str]] = [(s, d) for s, d, _ in SOLVER_IMPLEMENTATIONS]
    solver_labels: dict[tuple[str, str], str] = {
        (s, d): lbl for s, d, lbl in SOLVER_IMPLEMENTATIONS
    }
    baseline_solver, baseline_device = "MATLAB ode45", "cpu"

    n_values: list[int] = sorted(df["N"].unique().tolist())

    n_headers: list[str] = [f"N = {n:,}" for n in n_values]
    table_lines: list[str] = [
        "| Solver | " + " | ".join(n_headers) + " |",
        "|--------|" + "|".join(["--:" for _ in n_values]) + "|",
    ]

    times_grid: dict[tuple[str, str], dict[int, float | None]] = {}
    for solver, device in solver_keys:
        times_grid[(solver, device)] = {}
        for n in n_values:
            sub = df[(df["solver"] == solver) & (df["device"] == device) & (df["N"] == n)]
            times_grid[(solver, device)][n] = (
                float(sub["mean_time"].iloc[0]) if len(sub) > 0 else None
            )

    ranks_per_n: dict[int, dict[tuple[str, str], int]] = {}
    for n in n_values:
        available: dict[tuple[str, str], float] = {}
        for key in solver_keys:
            t = times_grid[key][n]
            if t is not None:
                available[key] = t
        sorted_keys = sorted(available, key=lambda k: available[k])
        ranks_per_n[n] = {k: i + 1 for i, k in enumerate(sorted_keys)}

    for solver, device in solver_keys:
        is_baseline = solver == baseline_solver and device == baseline_device
        row_parts: list[str] = [solver_labels[(solver, device)]]
        for n in n_values:
            t = times_grid[(solver, device)][n]
            if t is None:
                row_parts.append("—")
            else:
                baseline_t = times_grid.get((baseline_solver, baseline_device), {}).get(n)
                rank = ranks_per_n[n].get((solver, device), 0)
                row_parts.append(_fmt_solver_cell(t, baseline_t, rank, is_baseline))
        table_lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(table_lines)


def solver_matlab_speedup_table() -> str:
    """Render speedup vs MATLAB table from CSV data.

    Same layout as ``solver_comparison_table`` but only shows the speedup
    factor relative to MATLAB ode45 (no absolute times).

    :return: Markdown table string.
    """
    csv_path = SOLVER_COMPARISON_RESULTS_DIR / "solver_comparison.csv"

    if not csv_path.exists():
        return '!!! warning "Missing Data"\n    Solver comparison data not found.'

    df = pd.read_csv(csv_path)

    solver_keys: list[tuple[str, str]] = [(s, d) for s, d, _ in SOLVER_IMPLEMENTATIONS]
    solver_labels: dict[tuple[str, str], str] = {
        (s, d): lbl for s, d, lbl in SOLVER_IMPLEMENTATIONS
    }
    baseline_solver, baseline_device = "MATLAB ode45", "cpu"

    n_values: list[int] = sorted(df["N"].unique().tolist())

    n_headers: list[str] = [f"N = {n:,}" for n in n_values]
    table_lines: list[str] = [
        "| Solver | " + " | ".join(n_headers) + " |",
        "|--------|" + "|".join(["--:" for _ in n_values]) + "|",
    ]

    for solver, device in solver_keys:
        is_baseline = solver == baseline_solver and device == baseline_device
        row_parts: list[str] = [solver_labels[(solver, device)]]
        for n in n_values:
            sub = df[(df["solver"] == solver) & (df["device"] == device) & (df["N"] == n)]
            if len(sub) == 0:
                row_parts.append("—")
                continue
            t = float(sub["mean_time"].iloc[0])
            if is_baseline:
                row_parts.append("*baseline*")
            else:
                baseline_sub = df[
                    (df["solver"] == baseline_solver)
                    & (df["device"] == baseline_device)
                    & (df["N"] == n)
                ]
                if len(baseline_sub) > 0:
                    baseline_t = float(baseline_sub["mean_time"].iloc[0])
                    speedup = baseline_t / t
                    direction = "faster" if speedup > 1 else "slower"
                    row_parts.append(f"{speedup:.3f}× {direction}")
                else:
                    row_parts.append("—")
        table_lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(table_lines)


def benchmark_comparison_table() -> str:
    """Render benchmark comparison table from CSV data.

    Each non-MATLAB column shows time and speedup vs MATLAB, e.g. ``5.73s (2.8x)``.
    The fastest implementation per row is **bolded**. When the GPU wins, the best
    CPU-only option is also bolded so readers can identify the best no-GPU alternative.
    Prefers the all-tools CSV (generated with ``--all``) over the two-tool CSV.

    :return: Markdown table string.
    """
    csv_path_all = BENCHMARK_RESULTS_DIR / "end_to_end_comparison_all.csv"
    csv_path = BENCHMARK_RESULTS_DIR / "end_to_end_comparison.csv"

    if csv_path_all.exists():
        df = pd.read_csv(csv_path_all)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return '!!! warning "Missing Data"\n    Benchmark data not found. Run `bash scripts/generate_benchmark_plots.sh --all` to generate.'

    impl_order = ["MATLAB", "Python CPU", "Python CUDA", "Attractors.jl", "pynamicalsys"]
    present_impls = [impl for impl in impl_order if impl in df["implementation"].unique()]
    has_cuda = "Python CUDA" in present_impls

    table_lines: list[str] = [
        "| N | " + " | ".join(present_impls) + " |",
        "|--:|" + "|".join(["--:" for _ in present_impls]) + "|",
    ]

    cuda_won_any = False

    for n in sorted(df["N"].unique()):
        n_data = df[df["N"] == n]
        matlab_row = n_data[n_data["implementation"] == "MATLAB"]
        matlab_time = matlab_row["mean_time"].values[0] if len(matlab_row) > 0 else float("nan")

        # Collect times for present implementations
        row_times: dict[str, float] = {}
        for impl in present_impls:
            impl_row = n_data[n_data["implementation"] == impl]
            if len(impl_row) > 0:
                row_times[impl] = float(impl_row["mean_time"].values[0])

        winner = min(row_times, key=lambda x: row_times[x]) if row_times else None
        cuda_won = winner == "Python CUDA"
        if cuda_won:
            cuda_won_any = True
            # Best non-GPU, non-MATLAB option for readers without a GPU
            cpu_candidates = {
                k: v for k, v in row_times.items() if k not in ("Python CUDA", "MATLAB")
            }
            best_cpu = (
                min(cpu_candidates, key=lambda x: cpu_candidates[x]) if cpu_candidates else None
            )
        else:
            best_cpu = None

        cells: list[str] = [f"{n:,}"]
        for impl in present_impls:
            if impl not in row_times:
                cells.append("—")
                continue
            t = row_times[impl]
            if impl == "MATLAB":
                cell = f"{t:.2f}s"
            else:
                factor = matlab_time / t if t > 0 and not np.isnan(matlab_time) else float("nan")
                factor_str = f"{factor:.1f}×" if not np.isnan(factor) else "?"
                cell = f"{t:.2f}s ({factor_str})"

            if impl == winner or (cuda_won and impl == best_cpu):
                cell = f"**{cell}**"
            cells.append(cell)
        table_lines.append("| " + " | ".join(cells) + " |")

    result = "\n".join(table_lines)
    if cuda_won_any and has_cuda:
        result += (
            "\n\n*Bold marks the fastest per row. "
            "When the GPU wins, the best CPU-only option is also bolded — "
            "use it as the recommended alternative when no GPU is available.*"
        )
    return result


def benchmark_scaling_analysis() -> str:
    """Render scaling analysis summary from benchmark data, sorted fastest first.

    :return: Markdown summary string.
    """
    csv_path_all = BENCHMARK_RESULTS_DIR / "end_to_end_comparison_all.csv"
    csv_path = BENCHMARK_RESULTS_DIR / "end_to_end_comparison.csv"

    if csv_path_all.exists():
        df = pd.read_csv(csv_path_all)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        return '!!! warning "Missing Data"\n    Benchmark data not found.'

    impl_order = ["MATLAB", "Python CPU", "Python CUDA", "Attractors.jl", "pynamicalsys"]
    implementations = [impl for impl in impl_order if impl in df["implementation"].unique()]

    rows: list[dict] = []

    for impl in implementations:
        impl_data = df[df["implementation"] == impl].sort_values("N")
        if len(impl_data) < 3:
            continue

        n_vals = impl_data["N"].values.astype(float)
        t_vals = impl_data["mean_time"].values

        log_n = np.log(n_vals)
        log_t = np.log(t_vals)
        result = scipy_stats.linregress(log_n, log_t)
        alpha = result.slope
        alpha_ci = 1.96 * result.stderr
        r2 = result.rvalue**2

        if alpha < 0.15:
            complexity = "O(1)"
        elif abs(alpha - 1.0) < 0.15:
            complexity = "O(N)"
        elif abs(alpha - 2.0) < 0.15:
            complexity = "O(N²)"
        else:
            complexity = f"O(N^{alpha:.2f})"

        # Geometric mean of observed times as the sort key (lower = faster)
        geomean = float(np.exp(np.mean(np.log(t_vals))))

        rows.append(
            {
                "impl": impl,
                "complexity": complexity,
                "alpha": alpha,
                "alpha_ci": alpha_ci,
                "r2": r2,
                "geomean": geomean,
            }
        )

    rows.sort(key=lambda r: r["geomean"])

    results: list[str] = [
        "| Implementation | Scaling | Exponent α | R² |",
        "|----------------|---------|------------|-----|",
    ]
    for r in rows:
        results.append(
            f"| {r['impl']} | {r['complexity']} | {r['alpha']:.2f} ± {r['alpha_ci']:.2f} | {r['r2']:.3f} |"
        )

    return "\n".join(results)


def define_env(env: Any) -> None:
    """Define macros for mkdocs-macros-plugin.

    :param env: The macro environment.
    """
    env.macro(comparison_table, "comparison_table")
    env.macro(load_snippet, "load_snippet")
    env.macro(benchmark_comparison_table, "benchmark_comparison_table")
    env.macro(benchmark_scaling_analysis, "benchmark_scaling_analysis")
    env.macro(solver_comparison_table, "solver_comparison_table")
    env.macro(solver_matlab_speedup_table, "solver_matlab_speedup_table")
