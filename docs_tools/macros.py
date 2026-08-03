# pyright: basic
"""MkDocs macros for TorrStack documentation.

The macros render committed case-study, benchmark, and calculator evidence and
load source snippets without executing experiments during a site build.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportMissingTypeArgument=false

import ast
import html
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from griffe import Docstring, Parser
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "results"
PYTHON_SOURCE_ROOTS = (
    PROJECT_ROOT / "packages" / "pybasin" / "src",
    PROJECT_ROOT / "packages" / "tstorch" / "src",
    PROJECT_ROOT / "packages" / "case_studies" / "src",
)
TSTORCH_REPORT_STORE = PROJECT_ROOT / "packages" / "tstorch" / "reports" / "store.json"
TSTORCH_FEATURE_SPECS = (
    PROJECT_ROOT / "packages" / "tstorch" / "tests" / "utils" / "feature_specs.py"
)


def _format_bs_with_se(bs: float, se: float) -> str:
    """Format basin stability with standard error."""
    return f"{bs:.5f} ± {se:.5f}"


def _markdown_table_cell(value: str) -> str:
    """Normalize text for a Markdown table cell."""
    return value.replace("\n", " ").replace("|", "\\|").strip()


def _normalize_docstring_markdown(value: str) -> str:
    """Convert docstring inline-code markup into Markdown-friendly text."""
    return re.sub(r"\s+", " ", value.replace("``", "`")).strip()


def _split_default_from_description(description: str) -> tuple[str, str | None]:
    """Extract a trailing 'Defaults to ...' sentence from a field description."""
    match = re.search(r"\s+Defaults?\s+to\s+(.+)\.?$", description)
    if not match:
        return description, None
    default = match.group(1).rstrip(".").strip().strip("`")
    return description[: match.start()].rstrip("."), default


def _annotation_to_markdown(annotation: ast.expr) -> tuple[str, bool]:
    """Render a TypedDict field annotation and whether it is optional."""
    raw = ast.unparse(annotation)
    match = re.fullmatch(r"NotRequired\[(.+)\]", raw)
    if match:
        return match.group(1), True
    match = re.fullmatch(r"Required\[(.+)\]", raw)
    if match:
        return match.group(1), False
    return raw, False


def _resolve_python_object_source(dotted_path: str) -> tuple[Path, str]:
    """Resolve a dotted Python object path to a source file and class name."""
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(f"Expected a fully qualified class path, got {dotted_path!r}")

    relative_module = Path(*module_path.split(".")).with_suffix(".py")
    for root in PYTHON_SOURCE_ROOTS:
        candidate = root / relative_module
        if candidate.exists():
            return candidate, class_name
    raise FileNotFoundError(f"Could not find source for {dotted_path!r}")


def typed_dict_schema(dotted_path: str, heading_level: int = 3) -> str:
    """Render a compact Markdown schema for a TypedDict class.

    :param dotted_path: Fully qualified TypedDict class path.
    :param heading_level: Markdown heading level for the schema signature.
    :return: Markdown schema block.
    """
    source_path, class_name = _resolve_python_object_source(dotted_path)
    module_ast = ast.parse(source_path.read_text())
    class_node = next(
        node
        for node in module_ast.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )

    attr_descriptions: dict[str, str] = {}
    docstring = ast.get_docstring(class_node)
    if docstring:
        for section in Docstring(docstring).parse(Parser.google):
            if type(section).__name__ == "DocstringSectionAttributes":
                for attribute in section.value:
                    attr_descriptions[attribute.name] = _normalize_docstring_markdown(
                        attribute.description
                    )

    fields: list[tuple[str, str, bool, str, str | None]] = []
    for statement in class_node.body:
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        field_name = statement.target.id
        rendered_type, optional = _annotation_to_markdown(statement.annotation)
        description, default = _split_default_from_description(
            attr_descriptions.get(field_name, "")
        )
        if optional and default is None:
            default = "optional"
        fields.append((field_name, rendered_type, optional, description, default))

    has_default = any(default is not None for _, _, _, _, default in fields)
    heading = "#" * heading_level
    lines = [
        f'<a id="{dotted_path}"></a>',
        "",
        f"{heading} {class_name}(TypedDict):",
        "",
    ]
    if has_default:
        lines.extend([
            "| Field | Type | Default | Description |",
            "| --- | --- | --- | --- |",
        ])
        for field_name, rendered_type, _, description, default in fields:
            lines.append(
                f"| `{field_name}` | `{rendered_type}` | `{default or '-'}`"
                f" | {_markdown_table_cell(description)} |"
            )
    else:
        lines.extend([
            "| Field | Type | Description |",
            "| --- | --- | --- |",
        ])
        for field_name, rendered_type, _, description, _ in fields:
            lines.append(
                f"| `{field_name}` | `{rendered_type}` | {_markdown_table_cell(description)} |"
            )
    return "\n".join(lines)


def api_object_heading(title: str, dotted_path: str, heading_level: int = 2) -> str:
    """Render a human-readable API heading with a linked source object path."""
    heading = "#" * heading_level
    escaped_path = html.escape(dotted_path, quote=True)
    return "\n".join([
        f'<a id="{escaped_path}"></a>',
        f"{heading} {title}",
        (
            f'<p class="doc-source-path"><a href="#{escaped_path}">'
            f"<code>{escaped_path}</code></a></p>"
        ),
    ])


def _tstorch_benchmark_helpers() -> Any:
    module_name = "pybasin_docs_tstorch_bench_store"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = PROJECT_ROOT / "scripts" / "benchmark" / "tstorch_bench_store.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import benchmark renderer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_tstorch_domain_features() -> dict[str, list[str]]:
    tree = ast.parse(TSTORCH_FEATURE_SPECS.read_text())
    domain_features: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            if not isinstance(value_node, ast.List):
                continue
            features: list[str] = []
            for item in value_node.elts:
                if not isinstance(item, ast.Call):
                    continue
                if not isinstance(item.func, ast.Name) or item.func.id != "FeatureSpec":
                    continue
                if not item.args or not isinstance(item.args[0], ast.Constant):
                    continue
                if isinstance(item.args[0].value, str):
                    features.append(item.args[0].value)
            if features:
                domain_features[key_node.value] = features
    return domain_features


def _load_tstorch_tsfresh_support() -> dict[str, bool]:
    tree = ast.parse(TSTORCH_FEATURE_SPECS.read_text())
    support: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "FeatureSpec":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        feature_name = node.args[0].value
        if not isinstance(feature_name, str):
            continue
        supports_tsfresh = True
        for keyword in node.keywords:
            if keyword.arg != "supports_tsfresh":
                continue
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                supports_tsfresh = False
        support[feature_name] = supports_tsfresh
    return support


def _tstorch_domain_title(domain: str) -> str:
    return domain.replace("_", " ").title()


def _tstorch_domain_slug(domain: str) -> str:
    return domain.replace("_", "-")


def _tstorch_status_mark(value: bool | None) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "?"


def _tstorch_accuracy_mark(
    feature: str, value: bool | None, tsfresh_support: dict[str, bool]
) -> str:
    if value is None and tsfresh_support.get(feature) is False:
        return "n/a"
    return _tstorch_status_mark(value)


def _tstorch_aggregate_status(section: dict[str, Any] | None, key: str) -> bool | None:
    if not section:
        return None
    values = [bool(entry[key]) for entry in section.values()]
    return all(values) if values else None


def _tstorch_feature_statuses(
    feature_data: dict[str, Any],
) -> tuple[bool | None, bool | None, bool | None]:
    return (
        _tstorch_aggregate_status(feature_data.get("precision"), "passed"),
        _tstorch_aggregate_status(feature_data.get("compile"), "compiled"),
        _tstorch_aggregate_status(feature_data.get("export"), "exported"),
    )


_TSTORCH_BENCHMARK_SUMMARY_VARIANTS = {
    "🧊": {
        "single trajectory": "single_trajectory_direct_call_uncompiled",
        "batched trajectory": "batched_trajectories_proxy_call_uncompiled",
        "batched trajectory + args": (
            "batched_trajectories_proxy_call_uncompiled_batched_arguments_count10"
        ),
    },
    "🔥": {
        "single trajectory": "single_trajectory_direct_call_compiled",
        "batched trajectory": "batched_trajectories_proxy_call_compiled",
        "batched trajectory + args": (
            "batched_trajectories_proxy_call_compiled_batched_arguments_count10"
        ),
    },
}


def _tstorch_speed_cell(
    helpers: Any, performance: dict[str, Any], variant: str, backend: str
) -> tuple[str, bool]:
    tstorch_cell = performance.get(variant, {}).get(backend)
    tsfresh_cell, _ = helpers.tsfresh_cell(performance, variant)
    if tstorch_cell is None or tsfresh_cell is None:
        return "-", False
    tstorch_mean = tstorch_cell["mean_ms"]
    tsfresh_mean = tsfresh_cell["mean_ms"]
    if tstorch_mean <= 0 or tsfresh_mean <= 0:
        return "-", False
    if tstorch_mean < tsfresh_mean:
        return f"{_tstorch_format_speedup(tsfresh_mean / tstorch_mean)} faster", True
    return f"{_tstorch_format_speedup(tstorch_mean / tsfresh_mean)} slower", False


def _tstorch_format_speedup(value: float) -> str:
    return f"{value:.2f}x"


def _tstorch_benchmark_summary_table(
    helpers: Any, store: dict[str, Any], features: list[str], backend: str
) -> list[str]:
    lines = [
        f"#### `{backend}`",
        "",
        '<div class="tstorch-benchmark-summary-table" markdown="1">',
        "",
        "<table>",
        "<thead>",
        "<tr>",
        "<th>feature (cold/warm)</th>",
        "<th>single trajectory</th>",
        "<th>batched trajectory</th>",
        "<th>batched trajectory and args</th>",
        "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for feature in features:
        performance = store[feature].get("performance", {})
        for mode, variants in _TSTORCH_BENCHMARK_SUMMARY_VARIANTS.items():
            lines.append("<tr>")
            lines.append(f"<td><code>{html.escape(feature)}</code> {mode}</td>")
            cells = [
                _tstorch_speed_cell(helpers, performance, variants[column], backend)
                for column in variants
            ]
            for value, faster in cells:
                css_class = ' class="tstorch-faster-cell"' if faster else ""
                lines.append(f"<td{css_class}>{html.escape(value)}</td>")
            lines.append("</tr>")
    lines.extend([
        "</tbody>",
        "</table>",
        "",
        "</div>",
        "",
    ])
    return lines


def _tstorch_benchmark_summary(
    helpers: Any, store: dict[str, Any], features: list[str]
) -> list[str]:
    lines = [
        "### Benchmark Summary",
        "",
    ]
    lines.extend(_tstorch_benchmark_summary_table(helpers, store, features, "tstorch_cpu"))
    lines.extend(_tstorch_benchmark_summary_table(helpers, store, features, "tstorch_gpu"))
    lines.extend([
        '<p class="tstorch-benchmark-summary-hint">🧊 = uncompiled cold execution. 🔥 = compiled'
        " warmed execution. Values are tstorch speed relative to tsfresh; the one-round"
        " compile-cost row is excluded. The args column uses count=10.</p>",
        "",
    ])
    return lines


def _tstorch_benchmark_store() -> dict[str, Any]:
    helpers = _tstorch_benchmark_helpers()
    return helpers.load_store(TSTORCH_REPORT_STORE)


def _tstorch_benchmark_intro() -> list[str]:
    return [
        "## What is benchmarked",
        "",
        "These pages track tstorch feature calculators against tsfresh and against tracing"
        " requirements needed by `TorchFeatureExtractor` consumers.",
        "",
        "| Check | What it answers |",
        "| --- | --- |",
        "| Numerical accuracy | Does the tstorch feature match tsfresh on the canonical"
        " pybasin trajectories, including NaN behavior? |",
        "| `torch.compile(fullgraph=True)` | Can the feature trace as one full graph under"
        " `torch.compile`? |",
        "| `torch.export` | Can the feature be captured by `torch.export` with a dynamic batch"
        " dimension? |",
        "| Single-trajectory performance | How fast is a raw feature call on the canonical"
        " trajectories? |",
        "| Batched performance | How fast is a raw feature call or `TorchFeatureExtractor` call"
        " on 10,000 pendulum trajectories? |",
        "| Batched-argument performance | How performance changes when a parametrized feature"
        " receives 1, 5, or 10 parameter combinations in one extractor call. |",
        "",
        "A `?` means the result has not been recorded in `store.json` yet. Regenerate with"
        " `scripts/benchmark/benchmark-feature.sh --match <domain>[/<feature>]`.",
        "",
        "Numerical accuracy passes when `abs(diff) <= atol + rtol * abs(tsfresh)` and NaN"
        " positions match. The default tolerance is `atol=1e-4, rtol=1e-4`; feature rows"
        " show `-` for `atol` or `rtol` when that value uses the default.",
        "",
    ]


def tstorch_benchmark_overview() -> str:
    """Render the tstorch benchmark overview from the benchmark store."""
    helpers = _tstorch_benchmark_helpers()
    store = _tstorch_benchmark_store()
    domain_features = _load_tstorch_domain_features()
    tsfresh_support = _load_tstorch_tsfresh_support()
    lines = _tstorch_benchmark_intro()
    lines.extend(helpers.render_benchmark_inventory())
    lines.extend(helpers.render_performance_summary(store))
    lines.extend([
        "## Domains",
        "",
        "| domain | features | numerical accuracy | torch.compile | torch.export |",
        "| --- | ---: | --- | --- | --- |",
    ])
    for domain in sorted(domain_features):
        features = [feature for feature in domain_features[domain] if feature in store]
        precision_values: list[bool] = []
        compile_values: list[bool] = []
        export_values: list[bool] = []
        has_precision = has_compile = has_export = False
        for feature in features:
            precision, compiled, exported = _tstorch_feature_statuses(store[feature])
            if precision is not None or tsfresh_support.get(feature) is False:
                has_precision = True
                precision_values.append(True if precision is None else precision)
            if compiled is not None:
                has_compile = True
                compile_values.append(compiled)
            if exported is not None:
                has_export = True
                export_values.append(exported)
        lines.append(
            f"| [{_tstorch_domain_title(domain)}](details/{_tstorch_domain_slug(domain)}.md) | "
            f"{len(features)} | "
            f"{_tstorch_status_mark(all(precision_values) if has_precision else None)} | "
            f"{_tstorch_status_mark(all(compile_values) if has_compile else None)} | "
            f"{_tstorch_status_mark(all(export_values) if has_export else None)} |"
        )
    return "\n".join(lines)


def tstorch_benchmark_domain(domain: str) -> str:
    """Render one tstorch benchmark domain page from the benchmark store."""
    helpers = _tstorch_benchmark_helpers()
    store = _tstorch_benchmark_store()
    domain_features = _load_tstorch_domain_features()
    tsfresh_support = _load_tstorch_tsfresh_support()
    features = [feature for feature in domain_features.get(domain, []) if feature in store]
    lines = [
        "Benchmark definitions and data-set descriptions are on the"
        " [evaluation overview](../index.md).",
        "",
        "## Feature status",
        "",
        "| feature | numerical accuracy | torch.compile | torch.export |",
        "| --- | --- | --- | --- |",
    ]
    for feature in features:
        precision, compiled, exported = _tstorch_feature_statuses(store[feature])
        lines.append(
            f"| [`{feature}`](#{feature}) | "
            f"{_tstorch_accuracy_mark(feature, precision, tsfresh_support)} | "
            f"{_tstorch_status_mark(compiled)} | {_tstorch_status_mark(exported)} |"
        )
    lines.append("")
    lines.extend(_tstorch_benchmark_summary(helpers, store, features))

    for feature in features:
        lines.append(f"## `{feature}`")
        lines.append("")
        note = helpers.FEATURE_NOTES.get(feature)
        if note:
            lines.append(f"**Note:** {note}")
            lines.append("")
        precision = store[feature].get("precision")
        if precision:
            lines.extend(helpers.render_precision_table(precision))
            lines.append("")
        compile_section = store[feature].get("compile")
        if compile_section:
            lines.extend(helpers.render_compile_table(compile_section))
            lines.append("")
        export_section = store[feature].get("export")
        if export_section:
            lines.extend(helpers.render_export_table(export_section))
            lines.append("")
        performance = store[feature].get("performance")
        if performance:
            lines.extend(helpers.render_performance_table(performance))
            lines.append("")
    return "\n".join(lines)


def tstorch_numerical_summary() -> str:
    """Render numerical verification status by calculator domain."""
    store = _tstorch_benchmark_store()
    domain_features = _load_tstorch_domain_features()
    tsfresh_support = _load_tstorch_tsfresh_support()
    lines = [
        "The comparison uses the canonical pybasin trajectories and checks both finite"
        " values and NaN positions. A value passes when"
        " `abs(diff) <= atol + rtol * abs(tsfresh)`.",
        "",
        "| domain | recorded features | status |",
        "| --- | ---: | --- |",
    ]
    for domain in sorted(domain_features):
        features = [feature for feature in domain_features[domain] if feature in store]
        statuses: list[bool] = []
        for feature in features:
            precision, _, _ = _tstorch_feature_statuses(store[feature])
            if precision is not None or tsfresh_support.get(feature) is False:
                statuses.append(True if precision is None else precision)
        status = all(statuses) if statuses else None
        lines.append(
            f"| [{_tstorch_domain_title(domain)}](details/{_tstorch_domain_slug(domain)}.md)"
            f" | {len(statuses)} | {_tstorch_status_mark(status)} |"
        )
    return "\n".join(lines)


def tstorch_compile_export_summary() -> str:
    """Render compilation and export status by calculator domain."""
    store = _tstorch_benchmark_store()
    domain_features = _load_tstorch_domain_features()
    lines = [
        "Compilation is tested with `torch.compile(fullgraph=True)`. Export checks use a"
        " dynamic batch dimension so the captured program is useful beyond one fixed input.",
        "",
        "| domain | features | torch.compile | torch.export |",
        "| --- | ---: | --- | --- |",
    ]
    for domain in sorted(domain_features):
        features = [feature for feature in domain_features[domain] if feature in store]
        compile_values: list[bool] = []
        export_values: list[bool] = []
        for feature in features:
            _, compiled, exported = _tstorch_feature_statuses(store[feature])
            if compiled is not None:
                compile_values.append(compiled)
            if exported is not None:
                export_values.append(exported)
        lines.append(
            f"| [{_tstorch_domain_title(domain)}](details/{_tstorch_domain_slug(domain)}.md)"
            f" | {len(features)} |"
            f" {_tstorch_status_mark(all(compile_values) if compile_values else None)} |"
            f" {_tstorch_status_mark(all(export_values) if export_values else None)} |"
        )
    return "\n".join(lines)


def tstorch_performance_summary() -> str:
    """Render the cross-domain performance summary."""
    helpers = _tstorch_benchmark_helpers()
    return "\n".join(helpers.render_performance_summary(_tstorch_benchmark_store()))


def cpu_gpu_tabs(
    case_id: str,
    images: list[tuple[str, str]],
    gpu_images: list[tuple[str, str]] | None = None,
) -> str:
    """Render synced CPU/GPU result tabs for a case study.

    Uses Material for MkDocs' ``content.tabs.link`` feature: content tabs sharing the
    same title ("CPU"/"GPU") stay in sync with every other matching tab on the page --
    and site-wide, via localStorage -- so switching one switches all of them at once.

    :param case_id: The CPU ``comparison_table`` case id (e.g. ``"duffing_case1"``). The
        GPU variant is assumed to be ``f"{case_id}_gpu"``, matching the
        ``artifact_suffix="_gpu"`` convention the GPU-only integration tests write to.
    :param images: ``(alt_text, filename)`` pairs, filename relative to
        ``docs/assets/case_studies/`` for the CPU run. Unless ``gpu_images`` is given,
        GPU filenames are derived by inserting ``_gpu`` right after ``case_id`` in each
        CPU filename -- the CPU and GPU integration tests are written to request the
        same plots for a given case, so the two image lists normally stay in lockstep.
    :param gpu_images: Explicit override for the GPU tab's image list, for the rare case
        where it doesn't exactly mirror the CPU list (e.g. a plot failed to generate for
        that particular case). Defaults to the auto-derived list from ``images``.
    :return: Markdown for a synced CPU/GPU tabbed block.
    """
    gpu_case_id = f"{case_id}_gpu"

    def _render(tab_case_id: str, tab_images: list[tuple[str, str]]) -> str:
        lines = [comparison_table(tab_case_id), ""]
        for alt_text, filename in tab_images:
            lines.append(f"![{alt_text}](../assets/case_studies/{filename})")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _indent(text: str) -> str:
        return "\n".join(f"    {line}" if line else "" for line in text.split("\n"))

    if gpu_images is None:
        gpu_images = [
            (alt_text, filename.replace(case_id, gpu_case_id, 1)) for alt_text, filename in images
        ]

    cpu_block = _indent(_render(case_id, images))
    gpu_block = _indent(_render(gpu_case_id, gpu_images))

    return f'=== "CPU"\n\n{cpu_block}\n\n=== "GPU"\n\n{gpu_block}'


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

    :param spec: Specification in format "path/to/file.py::function_name". Path is
                 resolved relative to the project root and each entry in
                 ``PYTHON_SOURCE_ROOTS``, in that order.
    :return: Markdown-formatted code block with the extracted function.
    """
    try:
        file_path_str, func_name = spec.split("::")
    except ValueError:
        return f'!!! error "Invalid Format"\n    Expected format: `path/to/file.py::function_name`\n    Got: `{spec}`'

    file_path = None
    for root in (PROJECT_ROOT, *PYTHON_SOURCE_ROOTS):
        candidate = root / file_path_str
        if candidate.exists():
            file_path = candidate
            break

    if file_path is None:
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


BENCHMARK_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "end_to_end" / "results"
SOLVER_COMPARISON_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "solver_comparison" / "results"


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
            sub = cast(
                pd.DataFrame,
                df[(df["solver"] == solver) & (df["device"] == device) & (df["N"] == n)],
            )
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
            sub = cast(
                pd.DataFrame,
                df[(df["solver"] == solver) & (df["device"] == device) & (df["N"] == n)],
            )
            if len(sub) == 0:
                row_parts.append("—")
                continue
            t = float(sub["mean_time"].iloc[0])
            if is_baseline:
                row_parts.append("*baseline*")
            else:
                baseline_sub = cast(
                    pd.DataFrame,
                    df[
                        (df["solver"] == baseline_solver)
                        & (df["device"] == baseline_device)
                        & (df["N"] == n)
                    ],
                )
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
        n_data = cast(pd.DataFrame, df[df["N"] == n])
        matlab_row = cast(pd.DataFrame, n_data[n_data["implementation"] == "MATLAB"])
        matlab_time = matlab_row["mean_time"].values[0] if len(matlab_row) > 0 else float("nan")

        # Collect times for present implementations
        row_times: dict[str, float] = {}
        for impl in present_impls:
            impl_row = cast(pd.DataFrame, n_data[n_data["implementation"] == impl])
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
        impl_data = cast(pd.DataFrame, df[df["implementation"] == impl]).sort_values("N")
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

        rows.append({
            "impl": impl,
            "complexity": complexity,
            "alpha": alpha,
            "alpha_ci": alpha_ci,
            "r2": r2,
            "geomean": geomean,
        })

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
    env.macro(cpu_gpu_tabs, "cpu_gpu_tabs")
    env.macro(load_snippet, "load_snippet")
    env.macro(benchmark_comparison_table, "benchmark_comparison_table")
    env.macro(benchmark_scaling_analysis, "benchmark_scaling_analysis")
    env.macro(solver_comparison_table, "solver_comparison_table")
    env.macro(solver_matlab_speedup_table, "solver_matlab_speedup_table")
    env.macro(typed_dict_schema, "typed_dict_schema")
    env.macro(api_object_heading, "api_object_heading")
    env.macro(tstorch_benchmark_overview, "tstorch_benchmark_overview")
    env.macro(tstorch_benchmark_domain, "tstorch_benchmark_domain")
    env.macro(tstorch_numerical_summary, "tstorch_numerical_summary")
    env.macro(tstorch_compile_export_summary, "tstorch_compile_export_summary")
    env.macro(tstorch_performance_summary, "tstorch_performance_summary")
