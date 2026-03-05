"""MkDocs hooks for pybasin documentation.

Injects the installed package version into the site at build time so the
footer copyright and macro variable stay in sync with pyproject.toml
without manual updates.
"""

from importlib.metadata import PackageNotFoundError, version

from mkdocs.config.defaults import MkDocsConfig


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    try:
        pkg_version = version("pybasin")
    except PackageNotFoundError:
        pkg_version = "dev"

    config.copyright = f"Copyright &copy; 2025 Adrian Wix &mdash; pybasin v{pkg_version}"
    config.extra["version"] = pkg_version
    return config
