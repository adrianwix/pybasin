# pyright: basic
"""Thesis artifact utilities for CPSME-styled plots.

Provides color palette and export utilities following TU Berlin CPSME guidelines.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from cpsmehelper import export_figure, get_colors
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# CPSME color palette
_CPSME_COLORS: dict[str, str] = get_colors(format="hash")  # type: ignore[assignment]

# Semantic color mapping for basin stability plots
# Order optimized for maximum visual contrast with 5+ attractors
THESIS_PALETTE: list[str] = [
    _CPSME_COLORS["blue_3"],  # #457B9D - medium blue
    _CPSME_COLORS["red"],  # #E63946 - red
    _CPSME_COLORS["green"],  # #00b695 - teal green
    _CPSME_COLORS["grey_4"],  # #646464 - dark grey
    _CPSME_COLORS["blue_1"],  # #A8DADC - light blue
    _CPSME_COLORS["blue_2"],  # #008b9a - teal blue
    _CPSME_COLORS["blue_4"],  # #1D3557 - dark blue
    _CPSME_COLORS["grey_3"],  # #969696 - medium grey
]

# Lorenz-specific palette: red, blue, gray (sorted alphabetically: chaotic attractor 1, chaotic attractor 2, unbounded)
LORENZ_PALETTE: list[str] = [
    _CPSME_COLORS["red"],  # chaotic attractor 1
    _CPSME_COLORS["blue_3"],  # chaotic attractor 2
    _CPSME_COLORS["grey_4"],  # unbounded
]

# Export CPSME colors for direct use
CPSME: dict[str, str] = _CPSME_COLORS


def recolor_patches(ax: Axes, palette: list[str] | None = None) -> None:
    """Recolor bar patches on an axes using the CPSME palette.

    :param ax: Matplotlib axes containing bar patches.
    :param palette: Color list to cycle through. Defaults to THESIS_PALETTE.
    """
    if palette is None:
        palette = THESIS_PALETTE

    patches: list[Patch] = ax.patches  # type: ignore[assignment]
    for i, patch in enumerate(patches):
        patch.set_facecolor(palette[i % len(palette)])


def recolor_lines(ax: Axes, palette: list[str] | None = None) -> None:
    """Recolor line plots on an axes using the CPSME palette.

    :param ax: Matplotlib axes containing line artists.
    :param palette: Color list to cycle through. Defaults to THESIS_PALETTE.
    """
    if palette is None:
        palette = THESIS_PALETTE

    lines: list[Any] = ax.get_lines()
    for i, line in enumerate(lines):
        line.set_color(palette[i % len(palette)])
        line.set_markerfacecolor(palette[i % len(palette)])
        line.set_markeredgecolor(palette[i % len(palette)])


def recolor_scatters(ax: Axes, palette: list[str] | None = None) -> None:
    """Recolor scatter plot collections on an axes using the CPSME palette.

    :param ax: Matplotlib axes containing scatter collections.
    :param palette: Color list to cycle through. Defaults to THESIS_PALETTE.
    """
    if palette is None:
        palette = THESIS_PALETTE

    collections: list[PathCollection] = [c for c in ax.collections if isinstance(c, PathCollection)]
    for i, collection in enumerate(collections):
        collection.set_facecolor(palette[i % len(palette)])
        collection.set_edgecolor(palette[i % len(palette)])


def recolor_legend(ax: Axes, palette: list[str] | None = None) -> None:
    """Recolor legend handles on an axes using the CPSME palette.

    :param ax: Matplotlib axes containing a legend.
    :param palette: Color list to cycle through. Defaults to THESIS_PALETTE.
    """
    if palette is None:
        palette = THESIS_PALETTE

    legend = ax.get_legend()
    if legend is None:
        return

    for i, handle in enumerate(legend.legend_handles):  # type: ignore[union-attr]
        color = palette[i % len(palette)]
        # Handle different artist types
        if hasattr(handle, "set_facecolor"):
            handle.set_facecolor(color)  # type: ignore[union-attr]
        if hasattr(handle, "set_edgecolor"):
            handle.set_edgecolor(color)  # type: ignore[union-attr]
        if hasattr(handle, "set_color"):
            handle.set_color(color)  # type: ignore[union-attr]
        if hasattr(handle, "set_markerfacecolor"):
            handle.set_markerfacecolor(color)  # type: ignore[union-attr]
        if hasattr(handle, "set_markeredgecolor"):
            handle.set_markeredgecolor(color)  # type: ignore[union-attr]


def recolor_axes(ax: Axes, palette: list[str] | None = None) -> None:
    """Recolor all artist types on an axes using the CPSME palette.

    Handles patches (bars), lines, scatter collections, and legend handles.

    :param ax: Matplotlib axes to recolor.
    :param palette: Color list to cycle through. Defaults to THESIS_PALETTE.
    """
    if palette is None:
        palette = THESIS_PALETTE

    recolor_patches(ax, palette)
    recolor_lines(ax, palette)
    recolor_scatters(ax, palette)
    recolor_legend(ax, palette)


def recolor_figure(fig: Figure, palette: list[str] | None = None) -> None:
    """Recolor all axes in a figure using the CPSME palette.

    :param fig: Matplotlib figure to recolor.
    :param palette: Color list to cycle through. Defaults to THESIS_PALETTE.
    """
    if palette is None:
        palette = THESIS_PALETTE

    for ax in fig.axes:
        recolor_axes(ax, palette)  # type: ignore[arg-type]


def recolor_stacked_figure(fig: Figure, palette: list[str] | None = None) -> None:
    """Recolor stacked subplot figure with consistent colors across all axes.

    Unlike recolor_figure, this maintains a global color index so that each subplot
    gets a unique color from the palette. Useful for stacked trajectory plots where
    each subplot shows one trajectory.

    :param fig: Matplotlib figure with stacked subplots to recolor.
    :param palette: Color list to cycle through. Defaults to THESIS_PALETTE.
    """
    if palette is None:
        palette = THESIS_PALETTE

    for i, ax in enumerate(fig.axes):
        color = palette[i % len(palette)]
        lines: list[Any] = ax.get_lines()  # type: ignore[assignment]
        for line in lines:
            line.set_color(color)
            line.set_markerfacecolor(color)
            line.set_markeredgecolor(color)


def thesis_export(
    fig: Figure,
    name: str,
    save_dir: str | Path,
    style: str | None = None,
    width: float | None = None,
    height: float | None = None,
    resolution: int = 300,
) -> None:
    """Export a figure using CPSME export_figure utility.

    Wraps cpsmehelper.export_figure() with sensible defaults and closes the figure
    after export. If neither style nor dimensions are provided, uses figure's
    current size converted to cm.

    :param fig: Matplotlib figure to export.
    :param name: Output filename including extension (.png, .svg, .pdf, .tikz).
    :param save_dir: Directory to save the figure in.
    :param style: CPSME layout style (e.g., 'presentation_1x1'). Optional.
    :param width: Figure width in cm. Optional.
    :param height: Figure height in cm. Optional.
    :param resolution: DPI for raster formats. Default 300.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Build kwargs conditionally to avoid passing None values
    kwargs: dict[str, Any] = {
        "resolution": resolution,
    }
    if style is not None:
        kwargs["style"] = style
    elif width is None and height is None:
        # export_figure requires either style or dimensions
        # Convert figure size from inches to cm (1 inch = 2.54 cm)
        fig_width, fig_height = fig.get_size_inches()
        kwargs["width"] = fig_width * 2.54
        kwargs["height"] = fig_height * 2.54

    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height

    export_figure(
        fig,
        name=name,
        savedir=str(save_path),
        **kwargs,
    )
    plt.close(fig)
    print(f"  Written: {save_path / name}")


def thesis_save_plot(
    fig: Figure,
    path: Path,
    recolor: bool = True,
    palette: list[str] | None = None,
) -> None:
    """Save a figure with CPSME styling applied.

    Applies CPSME colors if requested, then exports using the CPSME export utility.

    :param fig: Matplotlib figure to save.
    :param path: Full output path including filename and extension.
    :param recolor: Whether to apply CPSME colors. Default True.
    :param palette: Color palette to use. Defaults to THESIS_PALETTE.
    """
    if recolor:
        recolor_figure(fig, palette)

    thesis_export(fig, name=path.name, save_dir=path.parent)
