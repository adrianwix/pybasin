"""Artifact generator for case study documentation.

Generates JSON comparison results and plot images from integration tests.
Outputs both PNG (for docs) and PDF (for thesis) using separate styling utilities.
"""

# pyright: basic

import json
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.matplotlib_study_plotter import MatplotlibStudyPlotter as DocsStudyPlotter
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter as DocsPlotter
from tests.integration.test_helpers import ComparisonResult, UnsupervisedComparisonResult

# Docs utilities and plotters (for PNG output)
from thesis_utils.docs_plots_utils import (
    LORENZ_PALETTE,
)
from thesis_utils.docs_plots_utils import (
    recolor_axes as docs_recolor_axes,
)
from thesis_utils.docs_plots_utils import (
    recolor_figure as docs_recolor_figure,
)
from thesis_utils.docs_plots_utils import (
    recolor_stacked_figure as docs_recolor_stacked_figure,
)
from thesis_utils.docs_plots_utils import (
    thesis_export as docs_export,
)
from thesis_utils.thesis_matplotlib_plotter import MatplotlibPlotter as ThesisPlotter

# Thesis utilities and plotters (for PDF output)
from thesis_utils.thesis_matplotlib_study_plotter import (
    MatplotlibStudyPlotter as ThesisStudyPlotter,
)
from thesis_utils.thesis_plots_utils import (
    LORENZ_PALETTE as THESIS_LORENZ_PALETTE,
)
from thesis_utils.thesis_plots_utils import (
    THESIS_HALF_HEIGHT_CM,
    THESIS_HALF_WIDTH_CM,
    configure_thesis_style,
    hide_line_attractors,
    hide_scatter_attractors,
    reformat_attractor_labels,
    remove_titles,
    rescale_artists,
    thesis_export,
)
from thesis_utils.thesis_plots_utils import (
    recolor_axes as thesis_recolor_axes,
)
from thesis_utils.thesis_plots_utils import (
    recolor_figure as thesis_recolor_figure,
)
from thesis_utils.thesis_plots_utils import (
    recolor_stacked_figure as thesis_recolor_stacked_figure,
)

configure_thesis_style()

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
RESULTS_DIR = ARTIFACTS_DIR / "results"
THESIS_CASE_STUDIES_DIR = ARTIFACTS_DIR / "case_studies"
DOCS_ASSETS_DIR = Path(__file__).parent.parent.parent / "docs" / "assets" / "case_studies"


def ensure_directories() -> None:
    """Create artifact directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_CASE_STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_single_point_artifacts(
    bse: BasinStabilityEstimator,
    comparison: ComparisonResult,
    trajectory_state: int = 0,
    trajectory_x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    trajectory_y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    phase_space_axes: tuple[int, int] | None = None,
    hidden_state_space_labels: list[str] | None = None,
    hidden_phase_space_labels: list[str] | None = None,
    trajectory_y_ticks: list[float] | None = None,
) -> None:
    """Generate artifacts for a single-point basin stability test.

    Writes comparison JSON to artifacts/results/, PNG images to docs/assets/,
    and PDF images to artifacts/case_studies/.

    :param bse: The BasinStabilityEstimator instance with results.
    :param comparison: The comparison result with validation metrics.
    :param trajectory_state: State variable index for trajectory plot.
    :param trajectory_x_limits: X-axis limits. Tuple applies to all, dict maps label to limits.
    :param trajectory_y_limits: Y-axis limits. Tuple applies to all, dict maps label to limits.
    :param phase_space_axes: Tuple (x_axis, y_axis) for 2D phase space plot. If None, skipped.
    :param hidden_state_space_labels: Attractor labels to hide from the state space legend.
    :param hidden_phase_space_labels: Attractor labels to hide from the phase space plot.
    :param trajectory_y_ticks: Explicit y-axis tick values applied to every trajectory subplot.
    """
    ensure_directories()

    system_name = comparison.system_name
    case_name = comparison.case_name
    prefix = f"{system_name}_{case_name}" if case_name else system_name

    palette = LORENZ_PALETTE if system_name == "lorenz" else None
    thesis_palette = THESIS_LORENZ_PALETTE if system_name == "lorenz" else None

    json_path = RESULTS_DIR / f"{prefix}_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)
    print(f"  Written: {json_path}")

    docs_plotter = DocsPlotter(bse)
    thesis_plotter = ThesisPlotter(bse)

    # Basin stability bars
    fig, ax = plt.subplots()
    docs_plotter.plot_basin_stability_bars(ax=ax)
    docs_recolor_axes(ax, palette)
    docs_export(fig, f"{prefix}_basin_stability.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots()
    thesis_plotter.plot_basin_stability_bars(ax=ax)
    thesis_recolor_axes(ax, thesis_palette)
    remove_titles(fig)
    reformat_attractor_labels(fig)
    thesis_export(
        fig,
        f"{prefix}_basin_stability.pdf",
        THESIS_CASE_STUDIES_DIR,
        width=THESIS_HALF_WIDTH_CM,
        height=THESIS_HALF_HEIGHT_CM,
    )

    # State space
    fig, ax = plt.subplots()
    docs_plotter.plot_state_space(ax=ax)
    docs_recolor_axes(ax, palette)
    docs_export(fig, f"{prefix}_state_space.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots()
    thesis_plotter.plot_state_space(ax=ax)
    thesis_recolor_axes(ax, thesis_palette)
    rescale_artists(fig, marker_size=0.3, alpha=1.0)
    remove_titles(fig)
    if hidden_state_space_labels:
        hide_scatter_attractors(fig, hidden_state_space_labels)
    reformat_attractor_labels(fig)
    thesis_export(
        fig,
        f"{prefix}_state_space.pdf",
        THESIS_CASE_STUDIES_DIR,
        width=THESIS_HALF_WIDTH_CM,
        height=THESIS_HALF_HEIGHT_CM,
    )

    # Feature space
    fig, ax = plt.subplots()
    docs_plotter.plot_feature_space(ax=ax)
    docs_recolor_axes(ax, palette)
    docs_export(fig, f"{prefix}_feature_space.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots()
    thesis_plotter.plot_feature_space(ax=ax)
    ax.set_xlabel(r"$\mathcal{X}_1$")
    ax.set_ylabel(r"$\mathcal{X}_2$")
    ax.legend(loc="lower right")
    thesis_recolor_axes(ax, thesis_palette)
    rescale_artists(fig, marker_size=1.0)
    remove_titles(fig)
    reformat_attractor_labels(fig)
    thesis_export(
        fig,
        f"{prefix}_feature_space.pdf",
        THESIS_CASE_STUDIES_DIR,
        width=THESIS_HALF_WIDTH_CM,
        height=THESIS_HALF_HEIGHT_CM,
    )

    # Trajectory plots
    if bse.template_integrator is not None:
        fig = docs_plotter.plot_templates_trajectories(
            plotted_var=trajectory_state,
            x_limits=trajectory_x_limits,
            y_limits=trajectory_y_limits,
        )
        docs_recolor_stacked_figure(fig, palette)
        docs_export(fig, f"{prefix}_trajectories.png", DOCS_ASSETS_DIR)

        fig = thesis_plotter.plot_templates_trajectories(
            plotted_var=trajectory_state,
            x_limits=trajectory_x_limits,
            y_limits=trajectory_y_limits,
        )
        thesis_recolor_stacked_figure(fig, thesis_palette)
        rescale_artists(fig, linewidth=1.0)
        remove_titles(fig)
        reformat_attractor_labels(fig)
        if trajectory_y_ticks is not None:
            for ax in fig.axes:
                ax.set_yticks(trajectory_y_ticks)
        thesis_export(
            fig,
            f"{prefix}_trajectories.pdf",
            THESIS_CASE_STUDIES_DIR,
            width=THESIS_HALF_WIDTH_CM,
            height=THESIS_HALF_HEIGHT_CM,
        )

    # Phase space plot
    if phase_space_axes is not None and bse.template_integrator is not None:
        fig = docs_plotter.plot_templates_phase_space(
            x_var=phase_space_axes[0],
            y_var=phase_space_axes[1],
        )
        docs_recolor_figure(fig, palette)
        docs_export(fig, f"{prefix}_phase_space.png", DOCS_ASSETS_DIR)

        fig = thesis_plotter.plot_templates_phase_space(
            x_var=phase_space_axes[0],
            y_var=phase_space_axes[1],
        )
        thesis_recolor_figure(fig, thesis_palette)
        rescale_artists(fig, linewidth=1.0)
        remove_titles(fig)
        if hidden_phase_space_labels:
            hide_line_attractors(fig, hidden_phase_space_labels)
        reformat_attractor_labels(fig)
        thesis_export(
            fig,
            f"{prefix}_phase_space.pdf",
            THESIS_CASE_STUDIES_DIR,
            width=THESIS_HALF_WIDTH_CM,
            height=THESIS_HALF_HEIGHT_CM,
        )


def generate_unsupervised_artifacts(
    bse: BasinStabilityEstimator,
    comparison: UnsupervisedComparisonResult,
    trajectory_state: int = 0,
    trajectory_x_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    trajectory_y_limits: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    phase_space_axes: tuple[int, int] | None = None,
    hidden_state_space_labels: list[str] | None = None,
    hidden_phase_space_labels: list[str] | None = None,
    trajectory_y_ticks: list[float] | None = None,
) -> None:
    """Generate artifacts for an unsupervised clustering test.

    Writes comparison JSON to artifacts/results/, PNG images to docs/assets/,
    and PDF images to artifacts/case_studies/.

    :param bse: The BasinStabilityEstimator instance with results.
    :param comparison: The unsupervised comparison result with cluster metrics.
    :param trajectory_state: State variable index for trajectory plot.
    :param trajectory_x_limits: X-axis limits. Tuple applies to all, dict maps label to limits.
    :param trajectory_y_limits: Y-axis limits. Tuple applies to all, dict maps label to limits.
    :param phase_space_axes: Tuple (x_axis, y_axis) for 2D phase space plot. If None, skipped.
    :param hidden_state_space_labels: Attractor labels to hide from the state space legend.
    :param hidden_phase_space_labels: Attractor labels to hide from the phase space plot.
    :param trajectory_y_ticks: Explicit y-axis tick values applied to every trajectory subplot.
    """
    ensure_directories()

    system_name = comparison.system_name
    case_name = comparison.case_name
    prefix = f"{system_name}_{case_name}" if case_name else system_name

    palette = LORENZ_PALETTE if system_name == "lorenz" else None
    thesis_palette = THESIS_LORENZ_PALETTE if system_name == "lorenz" else None

    json_path = RESULTS_DIR / f"{prefix}_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)
    print(f"  Written: {json_path}")

    docs_plotter = DocsPlotter(bse)
    thesis_plotter = ThesisPlotter(bse)

    # Basin stability bars
    fig, ax = plt.subplots()
    docs_plotter.plot_basin_stability_bars(ax=ax)
    docs_recolor_axes(ax, palette)
    docs_export(fig, f"{prefix}_basin_stability.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots()
    thesis_plotter.plot_basin_stability_bars(ax=ax)
    thesis_recolor_axes(ax, thesis_palette)
    remove_titles(fig)
    reformat_attractor_labels(fig)
    thesis_export(
        fig,
        f"{prefix}_basin_stability.pdf",
        THESIS_CASE_STUDIES_DIR,
        width=THESIS_HALF_WIDTH_CM,
        height=THESIS_HALF_HEIGHT_CM,
    )

    # State space
    fig, ax = plt.subplots()
    docs_plotter.plot_state_space(ax=ax)
    docs_recolor_axes(ax, palette)
    docs_export(fig, f"{prefix}_state_space.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots()
    thesis_plotter.plot_state_space(ax=ax)
    thesis_recolor_axes(ax, thesis_palette)
    rescale_artists(fig, marker_size=0.3, alpha=1.0)
    remove_titles(fig)
    if hidden_state_space_labels:
        hide_scatter_attractors(fig, hidden_state_space_labels)
    reformat_attractor_labels(fig)
    thesis_export(
        fig,
        f"{prefix}_state_space.pdf",
        THESIS_CASE_STUDIES_DIR,
        width=THESIS_HALF_WIDTH_CM,
        height=THESIS_HALF_HEIGHT_CM,
    )

    # Feature space
    fig, ax = plt.subplots()
    docs_plotter.plot_feature_space(ax=ax)
    docs_recolor_axes(ax, palette)
    docs_export(fig, f"{prefix}_feature_space.png", DOCS_ASSETS_DIR)

    fig, ax = plt.subplots()
    thesis_plotter.plot_feature_space(ax=ax)
    ax.set_xlabel(r"$\mathcal{X}_1$")
    ax.set_ylabel(r"$\mathcal{X}_2$")
    ax.legend(loc="lower right")
    thesis_recolor_axes(ax, thesis_palette)
    rescale_artists(fig, marker_size=1.0)
    remove_titles(fig)
    reformat_attractor_labels(fig)
    thesis_export(
        fig,
        f"{prefix}_feature_space.pdf",
        THESIS_CASE_STUDIES_DIR,
        width=THESIS_HALF_WIDTH_CM,
        height=THESIS_HALF_HEIGHT_CM,
    )

    # Trajectory plots
    if bse.template_integrator is not None:
        fig = docs_plotter.plot_templates_trajectories(
            plotted_var=trajectory_state,
            x_limits=trajectory_x_limits,
            y_limits=trajectory_y_limits,
        )
        docs_recolor_stacked_figure(fig, palette)
        docs_export(fig, f"{prefix}_trajectories.png", DOCS_ASSETS_DIR)

        fig = thesis_plotter.plot_templates_trajectories(
            plotted_var=trajectory_state,
            x_limits=trajectory_x_limits,
            y_limits=trajectory_y_limits,
        )
        thesis_recolor_stacked_figure(fig, thesis_palette)
        rescale_artists(fig, linewidth=1.0)
        remove_titles(fig)
        reformat_attractor_labels(fig)
        if trajectory_y_ticks is not None:
            for ax in fig.axes:
                ax.set_yticks(trajectory_y_ticks)
        thesis_export(
            fig,
            f"{prefix}_trajectories.pdf",
            THESIS_CASE_STUDIES_DIR,
            width=THESIS_HALF_WIDTH_CM,
            height=THESIS_HALF_HEIGHT_CM,
        )

    # Phase space plot
    if phase_space_axes is not None and bse.template_integrator is not None:
        fig = docs_plotter.plot_templates_phase_space(
            x_var=phase_space_axes[0],
            y_var=phase_space_axes[1],
        )
        docs_recolor_figure(fig, palette)
        docs_export(fig, f"{prefix}_phase_space.png", DOCS_ASSETS_DIR)

        fig = thesis_plotter.plot_templates_phase_space(
            x_var=phase_space_axes[0],
            y_var=phase_space_axes[1],
        )
        thesis_recolor_figure(fig, thesis_palette)
        rescale_artists(fig, linewidth=1.0)
        remove_titles(fig)
        if hidden_phase_space_labels:
            hide_line_attractors(fig, hidden_phase_space_labels)
        reformat_attractor_labels(fig)
        thesis_export(
            fig,
            f"{prefix}_phase_space.pdf",
            THESIS_CASE_STUDIES_DIR,
            width=THESIS_HALF_WIDTH_CM,
            height=THESIS_HALF_HEIGHT_CM,
        )


def generate_parameter_sweep_artifacts(
    bs_study: BasinStabilityStudy,
    comparisons: list[ComparisonResult],
    interval: Literal["linear", "log"] = "linear",
    skip_orbit_diagram: bool = False,
) -> None:
    """Generate artifacts for a parameter sweep basin stability test.

    Writes comparison JSON to artifacts/results/, PNG images to docs/assets/,
    and PDF images to artifacts/case_studies/.

    :param bs_study: The BasinStabilityStudy instance with results.
    :param comparisons: List of comparison results, one per parameter point.
    :param interval: x-axis scale — ``'linear'`` or ``'log'``.
    :param skip_orbit_diagram: Skip orbit diagram generation.
    """
    if not comparisons:
        return

    ensure_directories()

    system_name = comparisons[0].system_name
    case_name = comparisons[0].case_name
    prefix = f"{system_name}_{case_name}" if case_name else system_name

    palette = LORENZ_PALETTE if system_name == "lorenz" else None
    thesis_palette = THESIS_LORENZ_PALETTE if system_name == "lorenz" else None

    is_paper_validation = comparisons[0].paper_validation if comparisons else False

    data: dict[
        str,
        str
        | float
        | bool
        | list[dict[str, str | float | list[dict[str, str | float | bool]] | None]],
    ] = {
        "system_name": system_name,
        "case_name": case_name,
        "parameter_results": [c.to_dict() for c in comparisons],
    }
    if is_paper_validation:
        data["paper_validation"] = True

    json_path = RESULTS_DIR / f"{prefix}_comparison.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {json_path}")

    docs_plotter = DocsStudyPlotter(bs_study)
    thesis_plotter = ThesisStudyPlotter(bs_study)

    # Parameter stability plots (docs PNG)
    figs = docs_plotter.plot_parameter_stability(interval=interval)
    for fig in figs:
        param_name = fig.axes[0].get_xlabel() if fig.axes else ""  # type: ignore[union-attr]
        suffix = f"_{param_name}" if len(figs) > 1 else ""
        docs_recolor_figure(fig, palette)
        docs_export(fig, f"{prefix}_basin_stability_variation{suffix}.png", DOCS_ASSETS_DIR)

    # Parameter stability plots (thesis PDF)
    figs = thesis_plotter.plot_parameter_stability(interval=interval)
    for fig in figs:
        param_name = fig.axes[0].get_xlabel() if fig.axes else ""  # type: ignore[union-attr]
        suffix = f"_{param_name}" if len(figs) > 1 else ""
        thesis_recolor_figure(fig, thesis_palette)
        rescale_artists(fig, linewidth=1.0)
        remove_titles(fig)
        reformat_attractor_labels(fig)
        thesis_export(
            fig,
            f"{prefix}_basin_stability_variation{suffix}.pdf",
            THESIS_CASE_STUDIES_DIR,
            width=THESIS_HALF_WIDTH_CM,
            height=THESIS_HALF_HEIGHT_CM,
        )

    state_dim = bs_study.sampler.state_dim
    if skip_orbit_diagram:
        return
    if state_dim > 3:
        print(f"  Skipped orbit_diagram plot: state_dim={state_dim} > 3")
        return

    dof = [1] if state_dim >= 2 else [0]

    try:
        # Orbit diagram (docs PNG)
        figs = docs_plotter.plot_orbit_diagram(interval=interval, dof=dof)
        for fig in figs:
            param_name = fig.axes[0].get_xlabel() if fig.axes else ""  # type: ignore[union-attr]
            suffix = f"_{param_name}" if len(figs) > 1 else ""
            docs_recolor_figure(fig, palette)
            docs_export(fig, f"{prefix}_bifurcation_diagram{suffix}.png", DOCS_ASSETS_DIR)

        # Orbit diagram (thesis PDF)
        figs = thesis_plotter.plot_orbit_diagram(interval=interval, dof=dof)
        for fig in figs:
            param_name = fig.axes[0].get_xlabel() if fig.axes else ""  # type: ignore[union-attr]
            suffix = f"_{param_name}" if len(figs) > 1 else ""
            thesis_recolor_figure(fig, thesis_palette)
            rescale_artists(fig, marker_size=1.0)
            remove_titles(fig)
            reformat_attractor_labels(fig)
            thesis_export(
                fig,
                f"{prefix}_bifurcation_diagram{suffix}.pdf",
                THESIS_CASE_STUDIES_DIR,
                width=THESIS_HALF_WIDTH_CM,
                height=THESIS_HALF_HEIGHT_CM,
            )
    except (ValueError, KeyError) as e:
        print(f"  Skipped orbit_diagram plot: {e}")
