#!/usr/bin/env python3
"""Deterministic upsert + report generation for tstorch's feature status report.

Reads pytest-benchmark JSON (performance), utils/precision_recording JSON (numerical accuracy vs.
tsfresh), and utils/compile_recording JSON (torch.compile(fullgraph=True) safety) output, upserts
stats into a persistent per-feature store keyed by feature and (for performance) variant/backend
or (for precision/compile) param combo, and renders that store into a single markdown status
report per feature. This is the only source of truth for reported numbers - never hand-type a
stat into a doc.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TEST_NAME_RE = re.compile(
    r"^test_(?P<kind>"
    r"batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call"
    r"|batched_trajectories_proxy_call_compiled_cold_run"
    r"|batched_trajectories_proxy_call_compiled_batched_arguments"
    r"|batched_trajectories_proxy_call_uncompiled_batched_arguments"
    r"|batched_trajectories_proxy_call_compiled"
    r"|batched_trajectories_proxy_call_uncompiled"
    r"|single_trajectory_direct_call_uncompiled"
    r"|single_trajectory_direct_call_compiled"
    r")_[a-z0-9_]+?"
    r"\[(?:(?P<count>\d+)-)?(?P<backend>tstorch_cpu|tstorch_gpu|tsfresh)-(?P<feature>.+)\]$"
)

# Stable qualitative facts, not measurements - fine to hand-maintain here, unlike the
# numeric tables below, which are never hand-typed. Add an entry whenever a feature is
# truly batchable in tstorch but tsfresh's own calling convention does not share that
# computation - either because tsfresh calls it `fctype="simple"` or because the feature
# has no tsfresh equivalent at all.
FEATURE_NOTES: dict[str, str] = {
    "binned_entropy": (
        'tsfresh calls this once per bin count (`fctype="simple"`), recomputing the '
        "series range each time. tstorch shares min/max and NaN detection across all "
        "requested bin counts."
    ),
    "fourier_entropy": (
        'tsfresh calls this once per bin count (`fctype="simple"`) and recomputes the '
        "Welch spectrum for every call. tstorch computes one spectrum and reuses it "
        "across all requested bin counts."
    ),
    "cid_ce": (
        'tsfresh calls this once per normalization flag (`fctype="simple"`). tstorch '
        "computes the first-difference norm and standard deviation once, then selects "
        "the requested normalized or unnormalized columns."
    ),
    "count_above": (
        'tsfresh calls this once per threshold (`fctype="simple"`). tstorch broadcasts '
        "all requested thresholds through one tensor comparison and avoids a Python-level "
        "call per parameter."
    ),
    "count_below": (
        'tsfresh calls this once per threshold (`fctype="simple"`). tstorch broadcasts '
        "all requested thresholds through one tensor comparison and avoids a Python-level "
        "call per parameter."
    ),
    "count_in_range": (
        'tsfresh calls `range_count` once per interval (`fctype="simple"`). tstorch '
        "broadcasts all requested lower and upper bounds through one pair of tensor "
        "comparisons and avoids a Python-level call per parameter."
    ),
    "count_value": (
        'tsfresh calls `value_count` once per target value (`fctype="simple"`). tstorch '
        "broadcasts all requested values through one tensor comparison, including NaN "
        "matching, and avoids a Python-level call per parameter."
    ),
    "c3": (
        'tsfresh calls this once per lag (`fctype="simple"`). tstorch broadcasts all '
        "requested lags through one tensor operation and avoids a Python-level call per "
        "parameter."
    ),
    "time_reversal_asymmetry_statistic": (
        'tsfresh calls this once per lag (`fctype="simple"`). tstorch broadcasts all '
        "requested lags through one tensor operation and avoids a Python-level call per "
        "parameter."
    ),
    "autocorrelation": (
        'tsfresh calls this once *per lag* (`fctype="simple"`) with no sharing between '
        "calls - every lag re-derives the series mean/variance from scratch. tstorch's "
        "`_autocorrelation_kernel` computes the shared autocovariance-vs-lag array once "
        "and reads every requested lag off of it, so batching lags here is a genuine "
        "algorithmic improvement over tsfresh's own approach, not just a faster port of "
        "the same computation. See references/api-design.md."
    ),
    "quantile": (
        'tsfresh calls this once *per* q (`fctype="simple"`), independently re-sorting '
        "the series each time. tstorch's `_quantile` kernel sorts once and reads every "
        "requested q off the same sorted series via linear interpolation, so batching q's "
        "is a genuine algorithmic improvement over tsfresh's own approach."
    ),
    "mean_n_absolute_max": (
        'tsfresh calls this once *per* `number_of_maxima` (`fctype="simple"`), '
        "independently re-sorting the absolute values each time. tstorch's "
        "`_mean_n_absolute_max` kernel sorts once (via `topk`) and reads a cumulative mean "
        "at every requested `number_of_maxima` off that one sort, so batching is a genuine "
        "algorithmic improvement over tsfresh's own approach."
    ),
    "ratio_beyond_r_sigma": (
        'tsfresh calls this once *per* r (`fctype="simple"`), independently recomputing '
        "mean/std each time. tstorch's `_ratio_beyond_r_sigma` kernel computes mean/std "
        "once and broadcasts every requested r's threshold comparison against it, so "
        "batching r's is a genuine algorithmic improvement over tsfresh's own approach."
    ),
    "large_standard_deviation": (
        'tsfresh calls this once *per* r (`fctype="simple"`), independently recomputing '
        "the standard deviation and range for each threshold. tstorch computes std/range "
        "once and broadcasts them against every requested r."
    ),
    "change_quantiles": (
        'tsfresh calls this once *per* parameter combo (`fctype="simple"`), independently '
        "recomputing first differences and quantile binning each time. tstorch sorts once "
        "for every distinct quantile bound and broadcasts corridor masks across the "
        "requested parameter columns."
    ),
    "approximate_entropy": (
        "tsfresh flags this `high_comp_cost=True` and excludes it from "
        "`EfficientFCParameters` - it's an O(T^2) all-pairs Chebyshev-distance comparison, "
        'called once *per* (m, r) combo (`fctype="simple"`), independently rebuilding the '
        "whole distance matrix each time and processing one series at a time. tstorch's "
        "`ApproximateEntropy.__call__` broadcasts all requested `(m, r)` combinations "
        "through running Chebyshev-distance passes and vectorizes every (B, S) series in "
        "the batch. This improves both parameter handling and per-series execution rather "
        "than merely porting the row-at-a-time implementation."
    ),
    "sample_entropy": (
        "tsfresh flags this `high_comp_cost=True` and excludes it from "
        "`EfficientFCParameters` - it's an O(T^2) all-pairs Chebyshev-distance comparison "
        '(`fctype="simple"`), processed one series at a time. `SampleEntropy.__call__` '
        "vectorizes that comparison across every (B, S) series in the batch. Since the "
        "feature has no parameters, the gain comes from per-series batching rather than "
        "parameter sharing."
    ),
    "lempel_ziv_complexity": (
        "tsfresh calls this once per bin count and parses one symbolic series at a time. "
        "tstorch broadcasts symbolization across all requested bin counts, then sends the "
        "combined batch, state, and parameter rows through one registered phrase-count "
        "operator."
    ),
    "number_crossing_m": (
        'tsfresh calls this once *per* m (`fctype="simple"`), independently rebuilding '
        "the `x > m` mask each time. tstorch's `NumberCrossingM` class broadcasts "
        "every requested threshold into one comparison against `x`, so batching m's is a "
        "genuine algorithmic improvement over tsfresh's own approach."
    ),
    "number_cwt_peaks": (
        'tsfresh calls this once *per* n (`fctype="simple"`). tstorch places every '
        "requested Ricker width in one `conv1d` filter bank and shares that transform "
        "across n values. Ridge topology still has to be evaluated for each distinct "
        "width prefix because adding a scale can change how ridge lines are linked."
    ),
    "number_peaks": (
        'tsfresh calls this once *per* n (`fctype="simple"`), independently rebuilding '
        "the rolled neighbour comparisons for that n each time. tstorch's `NumberPeaks` "
        "class unfolds once at the largest requested n and reads every smaller n's peak "
        "mask off a cumulative logical AND of the same comparisons (beating neighbours up "
        "to distance d implies beating them up to any smaller distance), so batching n's "
        "is a genuine algorithmic improvement over tsfresh's own approach, not just a "
        "faster port of the same per-n computation."
    ),
}

VARIANT_ORDER = [
    "single_trajectory_direct_call_uncompiled",
    "single_trajectory_direct_call_compiled",
    "batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call",
    "batched_trajectories_proxy_call_compiled_cold_run",
    "batched_trajectories_proxy_call_compiled",
    "batched_trajectories_proxy_call_uncompiled",
    "batched_trajectories_proxy_call_uncompiled_batched_arguments",
    "batched_trajectories_proxy_call_compiled_batched_arguments",
]
VARIANT_LABEL = {
    "single_trajectory_direct_call_uncompiled": "single trajectory, direct_call, uncompiled",
    "single_trajectory_direct_call_compiled": "single trajectory, direct_call, compiled",
    "batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call": (
        "10,000 traj., tstorch direct_call uncompiled / tsfresh proxy_call"
    ),
    "batched_trajectories_proxy_call_compiled_cold_run": (
        "10,000 traj., proxy_call, compiled (cold run)"
    ),
    "batched_trajectories_proxy_call_compiled": "10,000 traj., proxy_call, compiled",
    "batched_trajectories_proxy_call_uncompiled": "10,000 traj., proxy_call, uncompiled",
    "batched_trajectories_proxy_call_uncompiled_batched_arguments": (
        "10,000 traj., proxy_call, uncompiled, batched arguments"
    ),
    "batched_trajectories_proxy_call_compiled_batched_arguments": (
        "10,000 traj., proxy_call, compiled, batched arguments"
    ),
}
BACKEND_ORDER = ["tstorch_cpu", "tstorch_gpu", "tsfresh"]
TSFRESH_ARGUMENT_BATCHING_LABEL = {
    "proxy_loops_per_argument": "proxy loops per argument",
    "calculator_gets_param_list_and_shares_work": "calculator gets param list and shares work",
    "calculator_gets_param_list_but_loops_arguments": (
        "calculator gets param list but loops arguments"
    ),
    "unsupported": "unsupported",
}
DEFAULT_NUMERICAL_ATOL = 1e-4
DEFAULT_NUMERICAL_RTOL = 1e-4

_VARIANT_COUNT_RE = re.compile(r"^(?P<base>.+)_count(?P<count>\d+)$")


def variant_base_and_count(variant: str) -> tuple[str, int | None]:
    match = _VARIANT_COUNT_RE.match(variant)
    if match is None:
        return variant, None
    return match.group("base"), int(match.group("count"))


def variant_label(variant: str) -> str:
    base, count = variant_base_and_count(variant)
    base_label = VARIANT_LABEL.get(base, base)
    return base_label if count is None else f"{base_label} (count={count})"


def format_variant_label(variant: str) -> str:
    return re.sub(r"\bcompiled\b", "**compiled**", variant_label(variant))


def variant_sort_key(variant: str) -> tuple[int, str, int]:
    base, count = variant_base_and_count(variant)
    order_idx = VARIANT_ORDER.index(base) if base in VARIANT_ORDER else len(VARIANT_ORDER)
    return (order_idx, base, -1 if count is None else count)


def is_current_variant(variant: str) -> bool:
    base, _ = variant_base_and_count(variant)
    return base in VARIANT_ORDER


def parse_benchmark_json(path: Path) -> list[dict[str, Any]]:
    """Extract (feature, variant, backend, stats) records from one pytest-benchmark JSON file."""
    payload: dict[str, Any] = json.loads(path.read_text())
    records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = payload.get("benchmarks", [])
    for entry in entries:
        match = TEST_NAME_RE.match(entry["name"])
        if match is None:
            continue
        count = match.group("count")
        variant = match.group("kind") + (f"_count{count}" if count else "")
        stats = entry["stats"]
        raw_extra_info: dict[str, Any] | None = entry.get("extra_info")
        extra_info: dict[str, Any] = raw_extra_info or {}
        records.append({
            "feature": match.group("feature"),
            "variant": variant,
            "backend": match.group("backend"),
            "mean_ms": stats["mean"] * 1000.0,
            "stddev_ms": stats["stddev"] * 1000.0,
            "min_ms": stats["min"] * 1000.0,
            "max_ms": stats["max"] * 1000.0,
            "median_ms": stats["median"] * 1000.0,
            "rounds": stats["rounds"],
            "extra_info": extra_info,
        })
    return records


def upsert_performance(
    records: list[dict[str, Any]], store: dict[str, Any], source: str
) -> dict[str, Any]:
    """Merge performance records into store, keyed by feature/variant/backend."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for record in records:
        feature_entry = store.setdefault(record["feature"], {})
        performance = feature_entry.setdefault("performance", {})
        variant_entry = performance.setdefault(record["variant"], {})
        variant_entry[record["backend"]] = {
            "mean_ms": record["mean_ms"],
            "stddev_ms": record["stddev_ms"],
            "min_ms": record["min_ms"],
            "max_ms": record["max_ms"],
            "median_ms": record["median_ms"],
            "rounds": record["rounds"],
            "updated_at": now,
            "source": source,
        }
        for key in (
            "tsfresh_argument_batching",
            "tsfresh_argument_count",
            "tsfresh_dispatcher_fctype",
            "tsfresh_argument_batching_source",
            "tsfresh_version",
            "tsfresh_measurement_key",
            "tsfresh_measurement_kind",
            "tsfresh_n_jobs",
        ):
            if key in record["extra_info"]:
                variant_entry[record["backend"]][key] = record["extra_info"][key]
    return store


def _precision_param_key(params: dict[str, Any], dtype: str = "") -> str:
    prefix = f"{dtype}:" if dtype else ""
    if not params:
        return f"{prefix}default"
    return f"{prefix}{tuple(sorted((key, repr(value)) for key, value in params.items()))!r}"


def precision_atol(entry: dict[str, Any]) -> float:
    value = entry.get("atol")
    return DEFAULT_NUMERICAL_ATOL if value is None else float(value)


def precision_rtol(entry: dict[str, Any]) -> float:
    value = entry.get("rtol")
    return DEFAULT_NUMERICAL_RTOL if value is None else float(value)


def format_precision_atol(entry: dict[str, Any]) -> str:
    atol = precision_atol(entry)
    return "-" if atol == DEFAULT_NUMERICAL_ATOL else f"`{atol:g}`"


def format_precision_rtol(entry: dict[str, Any]) -> str:
    rtol = precision_rtol(entry)
    return "-" if rtol == DEFAULT_NUMERICAL_RTOL else f"`{rtol:g}`"


def upsert_precision(
    records: list[dict[str, Any]], store: dict[str, Any], source: str
) -> dict[str, Any]:
    """Replace each tested feature's precision rows with the current recorded params."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for feature in {record["feature"] for record in records}:
        store.setdefault(feature, {})["precision"] = {}

    for record in records:
        feature_entry = store.setdefault(record["feature"], {})
        precision = feature_entry.setdefault("precision", {})
        dtype = record.get("dtype", "unknown")
        key = _precision_param_key(record["params"], dtype)
        precision[key] = {
            "params": record["params"],
            "dtype": dtype,
            "rows_compared": record["rows_compared"],
            "total_abs_diff": record["total_abs_diff"],
            "max_abs_diff": record["max_abs_diff"],
            "nan_mismatches": record["nan_mismatches"],
            "passed": record["passed"],
            "atol": DEFAULT_NUMERICAL_ATOL if record.get("atol") is None else record["atol"],
            "rtol": DEFAULT_NUMERICAL_RTOL if record.get("rtol") is None else record["rtol"],
            "test": record["test"],
            "updated_at": now,
            "source": source,
        }
    return store


def upsert_compile(
    records: list[dict[str, Any]], store: dict[str, Any], source: str
) -> dict[str, Any]:
    """Merge torch.compile(fullgraph=True) safety records into store, keyed by feature/params."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for record in records:
        feature_entry = store.setdefault(record["feature"], {})
        compile_section = feature_entry.setdefault("compile", {})
        key = _precision_param_key(record["params"])
        compile_section[key] = {
            "params": record["params"],
            "compiled": record["compiled"],
            "error": record["error"],
            "test": record["test"],
            "updated_at": now,
            "source": source,
        }
    return store


def upsert_export(
    records: list[dict[str, Any]], store: dict[str, Any], source: str
) -> dict[str, Any]:
    """Merge torch.export safety records into store, keyed by feature/params."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for record in records:
        feature_entry = store.setdefault(record["feature"], {})
        export_section = feature_entry.setdefault("export", {})
        key = _precision_param_key(record["params"])
        export_section[key] = {
            "params": record["params"],
            "exported": record["exported"],
            "error": record["error"],
            "test": record["test"],
            "updated_at": now,
            "source": source,
        }
    return store


def format_performance_cell(
    cell: dict[str, Any] | None, *, source_row: int | None = None, rank: int | None = None
) -> str:
    if cell is None:
        return "-"
    value = (
        f"{cell['mean_ms']:.3f} ± {cell['stddev_ms']:.3f} "
        f"({cell['min_ms']:.3f}–{cell['max_ms']:.3f}) [{cell['rounds']}]"
    )
    if source_row is not None:
        value += f" (Reused #{source_row})"
    if rank == 0:
        return f"**{value}**"
    if rank == 1:
        return f"<u>{value}</u>"
    return value


# Rows that do not measure a `tsfresh` timing themselves borrow the matching real tsfresh
# timing at render time. Count-suffixed row 8 variants borrow row 7 for the same count.
TSFRESH_BASELINE_VARIANT = {
    "single_trajectory_direct_call_compiled": "single_trajectory_direct_call_uncompiled",
    "batched_trajectories_proxy_call_compiled_cold_run": (
        "batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call"
    ),
    "batched_trajectories_proxy_call_compiled": (
        "batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call"
    ),
    "batched_trajectories_proxy_call_uncompiled": (
        "batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call"
    ),
    "batched_trajectories_proxy_call_compiled_batched_arguments": (
        "batched_trajectories_proxy_call_uncompiled_batched_arguments"
    ),
}
LEGACY_VARIANT_RENAMES = {
    "single_trajectory_uncompiled": "single_trajectory_direct_call_uncompiled",
    "batched_trajectories_direct": (
        "batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call"
    ),
    "batched_trajectories_compiled_cold_run": "batched_trajectories_proxy_call_compiled_cold_run",
    "batched_trajectories_compiled": "batched_trajectories_proxy_call_compiled",
    "batched_trajectories_compiled_batched_arguments": (
        "batched_trajectories_proxy_call_compiled_batched_arguments"
    ),
}
LEGACY_VARIANT_DROPS = {
    # Old row 2 used a warmed proxy call; new row 2 is a compiled raw Feature direct_call.
    "single_trajectory_compiled",
    # Old uncompiled proxy rows included an untimed warmup; the inventory now defines
    # uncompiled as no warmup.
    "batched_trajectories_uncompiled",
    "batched_trajectories_uncompiled_batched_arguments",
    "batched_direct_feature",
    "batched_feature",
    "batched_feature_after_compile",
    "batched_feature_no_compile",
    "single_feature",
}


def tsfresh_baseline_variant(variant: str) -> str:
    base, count = variant_base_and_count(variant)
    source_base = TSFRESH_BASELINE_VARIANT.get(base)
    if source_base is None:
        return variant
    return source_base if count is None else f"{source_base}_count{count}"


def normalize_performance_variants(store: dict[str, Any]) -> dict[str, Any]:
    for feature_name, raw_feature_data in store.items():
        feature_data: dict[str, Any] = raw_feature_data
        store[feature_name] = feature_data
        performance: dict[str, Any] | None = feature_data.get("performance")
        if performance is None:
            continue
        for variant in list(performance):
            base, count = variant_base_and_count(variant)
            target_base = LEGACY_VARIANT_RENAMES.get(base)
            if target_base is not None:
                target = target_base if count is None else f"{target_base}_count{count}"
                performance.setdefault(target, performance[variant])
                del performance[variant]
                continue
            if base in LEGACY_VARIANT_DROPS:
                del performance[variant]
    return store


def tsfresh_cell(
    performance: dict[str, Any], variant: str
) -> tuple[dict[str, Any] | None, str | None]:
    direct = performance.get(variant, {}).get("tsfresh")
    if direct is not None:
        return direct, None
    source_variant = tsfresh_baseline_variant(variant)
    if source_variant == variant:
        return None, None
    borrowed = performance.get(source_variant, {}).get("tsfresh")
    return borrowed, source_variant if borrowed is not None else None


def format_tsfresh_argument_batching(performance: dict[str, Any], variant: str) -> str:
    base, _ = variant_base_and_count(variant)
    if base not in {
        "batched_trajectories_proxy_call_uncompiled_batched_arguments",
        "batched_trajectories_proxy_call_compiled_batched_arguments",
    }:
        return "-"
    cell, source_variant = tsfresh_cell(performance, variant)
    if cell is None:
        return "-"
    batching = cell.get("tsfresh_argument_batching")
    if not isinstance(batching, str):
        return "-"
    label = TSFRESH_ARGUMENT_BATCHING_LABEL.get(batching)
    if label is None:
        return "-"
    if source_variant is not None:
        label += f" (reused from `{variant_label(source_variant)}`)"
    return label


def render_performance_table(performance: dict[str, Any]) -> list[str]:
    variants = sorted((v for v in performance if is_current_variant(v)), key=variant_sort_key)
    row_numbers = {variant: idx for idx, variant in enumerate(variants, start=1)}
    lines = [
        "### Performance",
        "",
        "`direct_call` constructs and calls the raw feature class. `proxy_call` uses the"
        " production extractor interface (`TorchFeatureExtractor` for tstorch,"
        " `extract_features` for tsfresh).",
        "",
    ]
    lines.extend([
        "`mean ± stddev (min–max) [rounds]`, in ms. Best timing per row is **bold**;"
        " second best is <u>underlined</u>:",
        "",
        '<div class="tstorch-performance-table" markdown="1">',
        "",
        "| # | variant | " + " | ".join(f"`{b}`" for b in BACKEND_ORDER) + " |",
        "| ---: | --- | " + " | ".join("---" for _ in BACKEND_ORDER) + " |",
    ])
    for variant in variants:
        label = format_variant_label(variant)
        cells: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        for backend in BACKEND_ORDER:
            if backend == "tsfresh":
                cells[backend] = tsfresh_cell(performance, variant)
            else:
                cells[backend] = (performance[variant].get(backend), None)
        ranked = sorted(
            (
                (backend, cell[0]["mean_ms"])
                for backend, cell in cells.items()
                if cell[0] is not None
            ),
            key=lambda item: item[1],
        )
        ranks = {backend: idx for idx, (backend, _) in enumerate(ranked[:2])}
        row: list[str] = []
        for backend in BACKEND_ORDER:
            cell, source_variant = cells[backend]
            source_row = row_numbers.get(source_variant) if source_variant is not None else None
            row.append(
                format_performance_cell(cell, source_row=source_row, rank=ranks.get(backend))
            )
        lines.append(f"| {row_numbers[variant]} | {label} | " + " | ".join(row) + " |")
    lines.extend(["", "</div>"])
    return lines


def render_benchmark_inventory() -> list[str]:
    return [
        "## Benchmark inventory",
        "",
        "| # | comparison | tstorch path | tsfresh path | scale | compiled state | parameter count |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
        "| 1 | single trajectory direct call | raw `Feature` construction and call | raw calculator "
        "through `execute_tsfresh_direct` | canonical trajectories | uncompiled | fixed |",
        "| 2 | single trajectory direct call, **compiled** | raw `Feature` call | row 1 timing "
        "reused | canonical trajectories | **compiled** and warmed | fixed |",
        "| 3 | batched trajectories direct call | raw `Feature` call | `extract_features` | "
        "10,000 pendulum trajectories | uncompiled | fixed |",
        "| 4 | batched extractor cold compile | fresh `TorchFeatureExtractor` | row 3 timing "
        "reused | 10,000 pendulum trajectories | **compiled** during measured round | fixed |",
        "| 5 | batched extractor warm **compiled** | reused `TorchFeatureExtractor` | row 3 "
        "timing reused | 10,000 pendulum trajectories | **compiled** and warmed | fixed |",
        "| 6 | batched extractor uncompiled | reused `TorchFeatureExtractor` | row 3 timing reused | "
        "10,000 pendulum trajectories | uncompiled, no warmup | fixed |",
        "| 7 | batched extractor arguments | reused `TorchFeatureExtractor` | `extract_features` "
        "with matched parameter count | 10,000 pendulum trajectories | uncompiled, no warmup | "
        "1 / 5 / 10 |",
        "| 8 | batched extractor arguments, **compiled** | reused `TorchFeatureExtractor` | row 7 "
        "timing reused for the same count | 10,000 pendulum trajectories | **compiled** and "
        "warmed | 1 / 5 / 10 |",
        "",
    ]


def render_performance_summary(store: dict[str, Any]) -> list[str]:
    """One row per performance variant: how many features tstorch is faster/slower than
    tsfresh on, for each of `tstorch_cpu`/`tstorch_gpu` - a quick "state of tstorch" read
    before diving into per-feature tables. Only counts a feature under a variant if both
    tsfresh (directly recorded or borrowed via `tsfresh_baseline_variant`) and the tstorch backend
    have a recorded mean; a feature missing one side (no tsfresh equivalent, or not yet benchmarked
    under that variant) is excluded from that variant's counts entirely, not counted either way.
    Lists the specific `tstorch_cpu`- and `tstorch_gpu`-slower features under the table, one list
    per backend per variant - `tstorch_gpu`'s single-trajectory numbers are
    dispatch/launch-overhead-dominated on the tiny per-call data that variant uses (see the note
    above the table), so treat that one list accordingly, but it's still reported for completeness.
    """
    lines = [
        "## Performance summary",
        "",
        "How many features tstorch beats tsfresh on, per comparison category (mean vs."
        " mean; only features with both a `tsfresh` and the given tstorch backend's"
        " recorded mean count toward that row. Rows without a native tsfresh axis borrow"
        " their matching real tsfresh row at render time, including the same-count row 7"
        " tsfresh value for row 8 batched-argument variants). GPU counts are frequently"
        " noise-dominated for the single-trajectory variant specifically (tiny per-call"
        " kernel-launch overhead dominates real compute there) - read the per-feature"
        " tables' min/max before treating a single-trajectory GPU comparison as decisive.",
        "",
        "| variant | tstorch_cpu faster | tstorch_cpu slower | tstorch_gpu faster | tstorch_gpu slower |",
        "| --- | --- | --- | --- | --- |",
    ]
    variants = sorted(
        {v for f in store for v in store[f].get("performance", {}) if is_current_variant(v)},
        key=variant_sort_key,
    )
    slower_by_variant: dict[str, dict[str, list[str]]] = {}
    for variant in variants:
        tsfresh_variant = tsfresh_baseline_variant(variant)
        counts = {"tstorch_cpu": [0, 0], "tstorch_gpu": [0, 0]}  # [faster, slower]
        slower: dict[str, list[str]] = {"tstorch_cpu": [], "tstorch_gpu": []}
        for feature, feature_data in store.items():
            performance = feature_data.get("performance", {})
            variant_perf = performance.get(variant)
            tsfresh_perf = performance.get(tsfresh_variant)
            if not variant_perf or not tsfresh_perf or "tsfresh" not in tsfresh_perf:
                continue
            tsfresh_mean = tsfresh_perf["tsfresh"]["mean_ms"]
            for backend in ("tstorch_cpu", "tstorch_gpu"):
                cell = variant_perf.get(backend)
                if cell is None:
                    continue
                idx = 1 if cell["mean_ms"] > tsfresh_mean else 0
                counts[backend][idx] += 1
                if idx == 1:
                    slower[backend].append(feature)
        label = format_variant_label(variant)
        lines.append(
            f"| {label} | {counts['tstorch_cpu'][0]} | {counts['tstorch_cpu'][1]} | "
            f"{counts['tstorch_gpu'][0]} | {counts['tstorch_gpu'][1]} |"
        )
        slower_by_variant[variant] = {b: sorted(names) for b, names in slower.items()}

    for backend in ("tstorch_cpu", "tstorch_gpu"):
        lines.append("")
        lines.append(f"`{backend}` slower than `tsfresh`, by variant:")
        lines.append("")
        for variant in variants:
            label = format_variant_label(variant)
            names_list = slower_by_variant[variant][backend]
            names = ", ".join(f"`{name}`" for name in names_list) if names_list else "none"
            lines.append(f"- {label}: {names}")
    lines.append("")
    return lines


def render_precision_table(precision: dict[str, Any]) -> list[str]:
    tests = sorted({entry["test"] for entry in precision.values()})
    lines = [
        "### Numerical accuracy vs. tsfresh",
        "",
    ]
    if len(tests) == 1:
        lines.extend([f"Test: `{tests[0]}`", ""])
    elif tests:
        lines.extend(["Tests:", "", *(f"- `{test}`" for test in tests), ""])
    lines.extend([
        "Pass equation: `abs(diff) <= atol + rtol * abs(tsfresh)`; NaN positions must match.",
        "",
        "| dtype | params | atol | rtol | rows | total abs diff | max abs diff | NaN mismatches | result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for key in sorted(precision):
        entry = precision[key]
        params = entry["params"]
        params_str = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items())) or "(none)"
        result = "pass" if entry["passed"] else "**FAIL**"
        lines.append(
            f"| `{entry.get('dtype', 'unknown')}` | `{params_str}` | "
            f"{format_precision_atol(entry)} | "
            f"{format_precision_rtol(entry)} | "
            f"{entry['rows_compared']} | {entry['total_abs_diff']:.3e} "
            f"| {entry['max_abs_diff']:.3e} | {entry['nan_mismatches']} | {result} |"
        )
    return lines


def render_compile_table(compile_section: dict[str, Any]) -> list[str]:
    tests = sorted({entry["test"] for entry in compile_section.values()})
    lines = [
        "### `torch.compile(fullgraph=True)` safety",
        "",
        'Checked with `options={"trace.enabled": True}`'
        " (https://docs.pytorch.org/docs/2.12/generated/torch.compile.html):",
        "",
    ]
    if len(tests) == 1:
        lines.extend([f"Test: `{tests[0]}`", ""])
    elif tests:
        lines.extend(["Tests:", "", *(f"- `{test}`" for test in tests), ""])
    lines.extend([
        "| params | compiles fullgraph | error |",
        "| --- | --- | --- |",
    ])
    for key in sorted(compile_section):
        entry = compile_section[key]
        params = entry["params"]
        params_str = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items())) or "(none)"
        result = "yes" if entry["compiled"] else "**NO**"
        error = f"`{entry['error']}`" if entry["error"] else "-"
        lines.append(f"| `{params_str}` | {result} | {error} |")
    return lines


def render_export_table(export_section: dict[str, Any]) -> list[str]:
    tests = sorted({entry["test"] for entry in export_section.values()})
    lines = [
        "### `torch.export` safety",
        "",
    ]
    if len(tests) == 1:
        lines.extend([f"Test: `{tests[0]}`", ""])
    elif tests:
        lines.extend(["Tests:", "", *(f"- `{test}`" for test in tests), ""])
    lines.extend([
        "| params | exports cleanly | error |",
        "| --- | --- | --- |",
    ])
    for key in sorted(export_section):
        entry = export_section[key]
        params = entry["params"]
        params_str = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items())) or "(none)"
        result = "yes" if entry["exported"] else "**NO**"
        error = f"`{entry['error']}`" if entry["error"] else "-"
        lines.append(f"| `{params_str}` | {result} | {error} |")
    return lines


def render_report(store: dict[str, Any]) -> str:
    features = sorted(store)
    lines = [
        "<!-- Generated by scripts/benchmark/tstorch_bench_store.py - do not hand-edit. -->",
        "",
        "# tstorch feature status report",
        "",
        "Regenerate with `scripts/benchmark/benchmark-feature.sh --match <domain>`."
        " Numbers here are parsed straight from pytest/pytest-benchmark JSON output - never"
        " hand-typed.",
        "",
    ]
    lines.extend(render_performance_summary(store))
    lines += [
        "## Trajectory data",
        "",
        "Single-trajectory and numerical-accuracy rows use"
        " `benchmarks/tstorch/cache/canonical_supervised_bounded.pt` - four of pybasin's own"
        " case-study ODE systems (`case_studies/pendulum`, `case_studies/duffing_oscillator`,"
        " `case_studies/lorenz`, `case_studies/friction`; exact params/initial conditions in"
        " `packages/tstorch/tests/utils/trajectory_builders.py::build_canonical_trajectories`),"
        " each integrated and filtered to its steady-state window:",
        "",
        "| case | trajectories | states | T |",
        "| --- | --- | --- | --- |",
        "| `pendulum` | 2 | 2 | 50 |",
        "| `duffing` | 5 | 2 | 5000 |",
        "| `lorenz` | 2 | 3 | 4000 |",
        "| `friction` | 2 | 2 | 500 |",
        "",
        "Deliberately varied shapes/series lengths, including a very short one (`pendulum`),"
        " so a performance rewrite is checked for correctness (and numerical stability) across"
        " more than one regime. Same data backs both the numerical-accuracy and single-trajectory"
        " performance rows, so a rewrite is checked for correctness against the exact data it's"
        " benchmarked on.",
        "",
        "10,000-trajectory rows use `benchmarks/tstorch/cache/pendulum_grid_10000.pt` - 10,000"
        " `pendulum` trajectories (same ODE system/params as the canonical `pendulum` case above)"
        " sampled on a grid over initial conditions via `pybasin.samplers.EvenlySpacedSampler`, for"
        " batched-extractor-scale timing (see"
        " `trajectory_builders.py::build_pendulum_grid_trajectories`).",
        "",
        "## Reading the tables",
        "",
        '`tstorch_cpu`/`tstorch_gpu` "compiled (cold run)" rows are `[1]` round on'
        " purpose, not under-sampled: `torch.compile` caches compiled kernels by code"
        ' structure + shape guards, not by object identity, so a second in-process "fresh'
        ' extractor" call silently hits that cache instead of measuring a second genuine '
        "cold compile.",
        "",
        "Numerical accuracy passes when `abs(diff) <= atol + rtol * abs(tsfresh)` and NaN"
        " positions match. The default tolerance is `atol=1e-4, rtol=1e-4`; feature rows"
        " show `-` for `atol` or `rtol` when that value uses the default.",
        "",
        "## Features",
        "",
    ]
    lines.extend(f"- [`{feature}`](#{feature})" for feature in features)
    lines.append("")

    for feature in features:
        lines.append(f"## `{feature}`")
        lines.append("")
        note = FEATURE_NOTES.get(feature)
        if note:
            lines.append(f"**Note:** {note}")
            lines.append("")
        precision = store[feature].get("precision")
        if precision:
            lines.extend(render_precision_table(precision))
            lines.append("")
        compile_section = store[feature].get("compile")
        if compile_section:
            lines.extend(render_compile_table(compile_section))
            lines.append("")
        export_section = store[feature].get("export")
        if export_section:
            lines.extend(render_export_table(export_section))
            lines.append("")
        performance = store[feature].get("performance")
        if performance:
            lines.extend(render_performance_table(performance))
            lines.append("")
    return "\n".join(lines)


def load_store(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_store(path: Path, store: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    perf_parser = sub.add_parser(
        "upsert-performance", help="Merge one pytest-benchmark JSON into the store"
    )
    perf_parser.add_argument("json_path", type=Path)
    perf_parser.add_argument("--store", type=Path, required=True)

    precision_parser = sub.add_parser(
        "upsert-precision", help="Merge one precision_recording JSON into the store"
    )
    precision_parser.add_argument("json_path", type=Path)
    precision_parser.add_argument("--store", type=Path, required=True)

    compile_parser = sub.add_parser(
        "upsert-compile", help="Merge one compile_recording JSON into the store"
    )
    compile_parser.add_argument("json_path", type=Path)
    compile_parser.add_argument("--store", type=Path, required=True)

    export_parser = sub.add_parser(
        "upsert-export", help="Merge one export_recording JSON into the store"
    )
    export_parser.add_argument("json_path", type=Path)
    export_parser.add_argument("--store", type=Path, required=True)

    render_parser = sub.add_parser("render", help="Regenerate the markdown report from the store")
    render_parser.add_argument("--store", type=Path, required=True)
    render_parser.add_argument("--report", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "upsert-performance":
        store = load_store(args.store)
        records = parse_benchmark_json(args.json_path)
        if not records:
            print(f"warning: no recognized benchmark entries in {args.json_path}", file=sys.stderr)
        upsert_performance(records, store, source=args.json_path.name)
        save_store(args.store, store)
        print(f"upserted {len(records)} performance record(s) into {args.store}")
    elif args.command == "upsert-precision":
        store = load_store(args.store)
        records = json.loads(args.json_path.read_text())
        if not records:
            print(f"warning: no recorded precision entries in {args.json_path}", file=sys.stderr)
        upsert_precision(records, store, source=args.json_path.name)
        save_store(args.store, store)
        print(f"upserted {len(records)} precision record(s) into {args.store}")
    elif args.command == "upsert-compile":
        store = load_store(args.store)
        records = json.loads(args.json_path.read_text())
        if not records:
            print(f"warning: no recorded compile entries in {args.json_path}", file=sys.stderr)
        upsert_compile(records, store, source=args.json_path.name)
        save_store(args.store, store)
        print(f"upserted {len(records)} compile record(s) into {args.store}")
    elif args.command == "upsert-export":
        store = load_store(args.store)
        records = json.loads(args.json_path.read_text())
        if not records:
            print(f"warning: no recorded export entries in {args.json_path}", file=sys.stderr)
        upsert_export(records, store, source=args.json_path.name)
        save_store(args.store, store)
        print(f"upserted {len(records)} export record(s) into {args.store}")
    elif args.command == "render":
        store = load_store(args.store)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(store))
        print(f"rendered {args.report}")


if __name__ == "__main__":
    main()
