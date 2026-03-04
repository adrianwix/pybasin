# pyright: basic
"""Thesis artifact utilities for CPSME-styled plots.

Provides color palette and export utilities following TU Berlin CPSME guidelines.
"""

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from cpsmehelper import export_figure, get_colors
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Thesis layout dimensions
# ---------------------------------------------------------------------------
# Textwidth measured from the compiled document: \the\textwidth = 418.25368pt
# 418.25368pt × 0.0352778 cm/pt = 14.755 cm
THESIS_TEXT_WIDTH_CM: float = 14.755

# Width for a single figure spanning the full text column
THESIS_FULL_WIDTH_CM: float = THESIS_TEXT_WIDTH_CM

# Width for two side-by-side figures (0.48\textwidth each, accounting for \hfill gap)
THESIS_HALF_WIDTH_CM: float = THESIS_TEXT_WIDTH_CM * 0.48

# Shared height for square-ish single plots and side-by-side pairs
# (golden ratio applied to half-width gives ~4.7 cm; 5.5 cm adds breathing room)
THESIS_SINGLE_HEIGHT_CM: float = THESIS_FULL_WIDTH_CM / 1.618
THESIS_HALF_HEIGHT_CM: float = THESIS_HALF_WIDTH_CM / 1.618

# Document base font size in pt (\documentclass{article} default)
_THESIS_FONT_SIZE_PT: float = 10.0


def configure_thesis_style() -> None:
    """Configure matplotlib rcParams to match the thesis document style.

    Sets the font family to TeX Gyre Heros (the OpenType Helvetica clone matching
    the document's phv/helvet font), the base font size to 10 pt (article class
    default), and disables TeX rendering so that the system font is used directly.
    Call this once at the start of any script that generates thesis figures.
    """
    plt.rcParams.update(  # type: ignore[misc]
        {
            "font.family": "sans-serif",
            # TeX Gyre Heros is the OpenType Helvetica clone used by LaTeX helvet/phv;
            # Helvetica and Arial as fallbacks if not installed
            "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": _THESIS_FONT_SIZE_PT,
            "axes.titlesize": _THESIS_FONT_SIZE_PT,
            "axes.labelsize": _THESIS_FONT_SIZE_PT,
            "xtick.labelsize": _THESIS_FONT_SIZE_PT - 1,
            "ytick.labelsize": _THESIS_FONT_SIZE_PT - 1,
            "legend.fontsize": _THESIS_FONT_SIZE_PT - 1,
            "figure.titlesize": _THESIS_FONT_SIZE_PT,
            # Disable TeX so matplotlib uses the system font directly
            "text.usetex": False,
        }
    )


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


def remove_titles(fig: Figure) -> None:
    """Remove titles from all axes and the figure suptitle.

    Figures exported for thesis use figure captions instead of plot titles.

    :param fig: Matplotlib figure whose titles will be cleared.
    """
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")  # type: ignore[union-attr]


def hide_scatter_attractors(fig: Figure, hidden_labels: list[str]) -> None:
    """Remove scatter collections for specific attractors from all axes in a figure.

    Removes the PathCollection artists whose label matches any entry in
    ``hidden_labels``, then rebuilds the legend without those entries.
    Used to hide attractor data points from state space scatter plots.

    :param fig: Matplotlib figure to modify.
    :param hidden_labels: Attractor labels whose data points should not be shown.
    """
    hidden_set = set(hidden_labels)
    for ax in fig.axes:
        to_remove = [
            c
            for c in ax.collections
            if isinstance(c, PathCollection) and c.get_label() in hidden_set
        ]
        for collection in to_remove:
            collection.remove()
        # Rebuild legend from remaining labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            if handles:
                ax.legend(handles, labels)  # type: ignore[misc]
            else:
                ax.get_legend().remove()  # type: ignore[union-attr]


def hide_line_attractors(fig: Figure, hidden_labels: list[str]) -> None:
    """Remove Line2D artists for specific attractors from all axes in a figure.

    Removes Line2D artists whose label matches any entry in ``hidden_labels``,
    then rebuilds the legend without those entries. Used to hide trajectories
    (e.g. unbounded attractors) from phase space plots.

    :param fig: Matplotlib figure to modify.
    :param hidden_labels: Attractor labels whose lines should not be shown.
    """
    hidden_set = set(hidden_labels)
    for ax in fig.axes:
        to_remove = [line for line in ax.get_lines() if line.get_label() in hidden_set]
        for line in to_remove:
            line.remove()
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            if handles:
                ax.legend(handles, labels)  # type: ignore[misc]
            else:
                ax.get_legend().remove()  # type: ignore[union-attr]


def reformat_attractor_labels(fig: Figure) -> None:
    """Rewrite attractor legend labels to proper notation.

    - ``y<n>`` → ``$\\bar{y}_{n}$``
    - ``chaotic attractor <n>`` → ``$\\bar{y}_{n}$ chaos``

    :param fig: Matplotlib figure whose legends will be updated.
    """
    pattern_y = re.compile(r"^y(\d+)$")
    pattern_chaos = re.compile(r"^chaotic attractor (\d+)$")
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        for text in legend.get_texts():
            t = text.get_text()
            m = pattern_y.match(t)
            if m:
                text.set_text(f"$\\bar{{y}}_{{{m.group(1)}}}$")
                continue
            m = pattern_chaos.match(t)
            if m:
                text.set_text(f"$\\bar{{y}}_{{{m.group(1)}}}$ chaos")


def rescale_artists(
    fig: Figure,
    marker_size: float = 1.0,
    linewidth: float = 1.0,
    alpha: float | None = None,
    legend_marker_size: float = 6.0,
) -> None:
    """Scale down scatter marker sizes and line widths on all axes in a figure.

    Plotters author figures at a generic size; this function adjusts artist
    dimensions so they remain visually appropriate after the figure is resized
    to thesis dimensions.

    :param fig: Matplotlib figure whose artists will be rescaled.
    :param marker_size: Scatter marker size in points² to apply to all PathCollections.
    :param linewidth: Line width in points to apply to all Line2D artists.
    :param alpha: If provided, overrides the alpha (opacity) of all PathCollections.
    :param legend_marker_size: Scatter marker size in points² to apply to legend
        PathCollection handles, giving a consistently visible dot regardless of
        how small the plot markers are.
    """
    for ax in fig.axes:
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_sizes([marker_size])
                if alpha is not None:
                    collection.set_alpha(alpha)
        for line in ax.get_lines():
            line.set_linewidth(linewidth)
        legend = ax.get_legend()
        if legend is not None:
            for handle in legend.legend_handles:
                if isinstance(handle, PathCollection):
                    handle.set_sizes([legend_marker_size])


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

    # export_figure requires either a valid style or both width AND height.
    # When only one dimension is given, derive the other from the figure's aspect ratio.
    fig_w_in, fig_h_in = fig.get_size_inches()
    fig_aspect = fig_h_in / fig_w_in  # height / width

    resolved_width = width
    resolved_height = height

    if resolved_width is None and resolved_height is None:
        resolved_width = fig_w_in * 2.54
        resolved_height = fig_h_in * 2.54
    elif resolved_width is not None and resolved_height is None:
        resolved_height = resolved_width * fig_aspect
    elif resolved_height is not None and resolved_width is None:
        resolved_width = resolved_height / fig_aspect

    # Pre-resize the figure to the target dimensions so that when
    # export_figure calls fig.set_size_inches with the same values,
    # fonts and line widths are not scaled down from their configured sizes.
    assert resolved_width is not None
    assert resolved_height is not None
    fig.set_size_inches(resolved_width / 2.54, resolved_height / 2.54)

    kwargs: dict[str, Any] = {
        "resolution": resolution,
    }
    if style is not None:
        kwargs["style"] = style
    else:
        kwargs["width"] = resolved_width
        kwargs["height"] = resolved_height

    export_figure(
        fig,
        name=name,
        savedir=str(save_path),
        **kwargs,
    )
    print(f"  Written: {save_path / name}")

    stem, ext = Path(name).stem, Path(name).suffix
    if ext.lower() == ".png":
        pdf_name = f"{stem}.pdf"
        export_figure(
            fig,
            name=pdf_name,
            savedir=str(save_path),
            **kwargs,
        )
        print(f"  Written: {save_path / pdf_name}")

    plt.close(fig)


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
