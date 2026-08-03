"""MkDocs hooks for TorrStack documentation.

Package builds include the installed distribution version in their footer.
"""

import html
import os
import re
from importlib.metadata import PackageNotFoundError, version

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    config_path = os.path.normpath(str(config.config_file_path)).replace(os.sep, "/")
    if "packages/tstorch/" in config_path:
        distribution = "tstorch"
    elif "packages/pybasin/" in config_path:
        distribution = "pybasin"
    else:
        distribution = None

    if distribution is None:
        config.copyright = "Copyright &copy; 2025 Adrian Wix &mdash; TorrStack"
        return config

    try:
        pkg_version = version(distribution)
    except PackageNotFoundError:
        pkg_version = "dev"

    config.copyright = f"Copyright &copy; 2025 Adrian Wix &mdash; {distribution} v{pkg_version}"
    config.extra["version"] = pkg_version
    return config


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
