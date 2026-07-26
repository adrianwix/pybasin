"""MkDocs hooks for pybasin documentation.

Injects the installed package version into the site at build time so the
footer copyright and macro variable stay in sync with pyproject.toml
without manual updates.
"""

import html
import re
from importlib.metadata import PackageNotFoundError, version

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

TSTORCH_BENCHMARK_DOMAINS = {
    "advanced": "Advanced",
    "autocorrelation": "Autocorrelation",
    "boolean": "Boolean",
    "change": "Change",
    "count": "Count",
    "dynamical": "Dynamical",
    "entropy_complexity": "Entropy Complexity",
    "frequency": "Frequency",
    "location": "Location",
    "pattern": "Pattern",
    "reocurrance": "Reocurrance",
    "statistical": "Statistical",
    "trend": "Trend",
}


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    try:
        pkg_version = version("pybasin")
    except PackageNotFoundError:
        pkg_version = "dev"

    config.copyright = f"Copyright &copy; 2025 Adrian Wix &mdash; pybasin v{pkg_version}"
    config.extra["version"] = pkg_version
    return config


def _domain_slug(domain: str) -> str:
    return domain.replace("_", "-")


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add tstorch benchmark pages rendered from the JSON benchmark store."""
    open_expr = "{" + "{"
    close_expr = "}" + "}"
    files.append(
        File.generated(
            config,
            "tstorch/benchmarks/index.md",
            content=(
                f"# tstorch benchmarks\n\n{open_expr} tstorch_benchmark_overview() {close_expr}\n"
            ),
        )
    )
    for domain, title in TSTORCH_BENCHMARK_DOMAINS.items():
        files.append(
            File.generated(
                config,
                f"tstorch/benchmarks/{_domain_slug(domain)}.md",
                content=(
                    f"# {title} benchmarks\n\n"
                    f"{open_expr} tstorch_benchmark_domain({domain!r}) {close_expr}\n"
                ),
            )
        )
    return files


def on_page_content(html_content: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Link rendered type names to manual schema anchors on the same page."""

    def replace_type_span(match: re.Match[str]) -> str:
        dotted_path = html.unescape(match.group("title"))
        label = match.group("label")
        if f'id="{dotted_path}"' not in html_content:
            return match.group(0)
        escaped_path = html.escape(dotted_path, quote=True)
        return (
            f'<a class="autorefs autorefs-internal" title="{escaped_path}" '
            f'href="#{escaped_path}">{label}</a>'
        )

    return re.sub(
        r'<span title="(?P<title>[^"]+Params)">(?P<label>[^<]+Params)</span>',
        replace_type_span,
        html_content,
    )
