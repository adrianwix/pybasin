"""AIO Feature Space scatter plot page with trajectory inspection."""

import logging
from collections.abc import Sequence
from typing import Any, cast

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from dash import (
    MATCH,
    Input,
    NoUpdate,
    Output,
    State,
    callback,  # pyright: ignore[reportUnknownVariableType]
    ctx,
    dcc,
    html,
    no_update,
)

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.trajectory_modal_aio import TrajectoryModalAIO
from pybasin.plotters.interactive_plotter.utils import get_color
from pybasin.plotters.types import FeatureSpaceOptions, filter_by_include_exclude

logger = logging.getLogger(__name__)


class FeatureSpaceAIO(BseBasePageAIO):
    """
    AIO component for feature space scatter plot with trajectory inspection.

    Supports dynamic feature axis selection with per-state feature labels.
    Embeds TrajectoryModalAIO for click-to-inspect functionality.
    """

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
        options: FeatureSpaceOptions | None = None,
    ):
        """
        Initialize feature space AIO component.

        :param bse: Basin stability estimator with computed results.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to labels.
        :param options: Feature space plot configuration options.
        """
        super().__init__(bse, aio_id, state_labels)
        self.options = options or {}
        FeatureSpaceAIO._instances[aio_id] = self
        self.trajectory_modal = TrajectoryModalAIO(bse, aio_id, state_labels)

    def get_feature_options(self, use_filtered: bool = True) -> list[dict[str, str]]:
        """Get dropdown options for feature selection with human-readable labels."""
        if self.bse.solution is None:
            return [{"value": "0", "label": "Feature 0"}]

        # Determine which feature names to use
        feature_names: list[str] | None = None
        if use_filtered:
            feature_names = self.bse.solution.feature_names
        else:
            feature_names = self.bse.solution.extracted_feature_names

        if feature_names is None:
            logger.warning(
                "No %s feature names available. Filtered names: %s, Extracted names: %s",
                "filtered" if use_filtered else "extracted",
                self.bse.solution.feature_names is not None,
                self.bse.solution.extracted_feature_names is not None,
            )
            return [{"value": "0", "label": "Feature 0"}]

        options: list[dict[str, str]] = []
        for idx, raw_name in enumerate(feature_names):
            parts = raw_name.split("__", 1)
            if len(parts) == 2 and parts[0].startswith("state_"):
                try:
                    state_idx = int(parts[0].replace("state_", ""))
                    state_label = self.get_state_label(state_idx)
                    feature_name = parts[1]
                    display_label = f"{state_label} - {feature_name}"
                except ValueError:
                    display_label = raw_name
            else:
                display_label = raw_name

            options.append({"value": str(idx), "label": display_label})

        logger.debug("Generated %d feature options (use_filtered=%s)", len(options), use_filtered)
        return options if options else [{"value": "0", "label": "Feature 0"}]

    def get_all_labels(self) -> list[str]:
        """Get all unique labels as strings."""
        if self.bse.solution is None or self.bse.solution.labels is None:
            return []
        labels_as_str = [str(lbl) for lbl in self.bse.solution.labels]
        unique_labels = np.unique(labels_as_str)
        return [str(label) for label in unique_labels]

    def render(self) -> html.Div:
        """Render complete page layout with controls, plot, and modal."""
        use_filtered = self.options.get("use_filtered", True)
        feature_options = self.get_feature_options(use_filtered=use_filtered)
        all_labels = self.get_all_labels()
        selected_labels = filter_by_include_exclude(
            all_labels,
            self.options.get("include_labels"),
            self.options.get("exclude_labels"),
        )

        feature_select_data = cast(Sequence[str], feature_options)
        x_axis = self.options.get("x_axis", 0)
        y_axis = self.options.get("y_axis", 1)

        show_feature_switch = (
            self.bse.solution is not None
            and self.bse.solution.extracted_features is not None
            and self.bse.solution.features is not None
            and self.bse.feature_selector is not None
        )

        return html.Div([
            dmc.Grid(
                [
                    dmc.GridCol(
                        [
                            dmc.Flex(
                                [
                                    dmc.Select(
                                        id=aio_id("FeatureSpace", self.aio_id, "x-select"),
                                        label="X Axis",
                                        data=feature_select_data,
                                        value=str(x_axis),
                                    ),
                                    dmc.Select(
                                        id=aio_id("FeatureSpace", self.aio_id, "y-select"),
                                        label="Y Axis",
                                        data=feature_select_data,
                                        value=str(y_axis) if y_axis is not None else "1",
                                    ),
                                    dmc.MultiSelect(
                                        id=aio_id("FeatureSpace", self.aio_id, "label-select"),
                                        label="Labels",
                                        data=all_labels,
                                        value=selected_labels,
                                    ),
                                    dmc.Stack(
                                        [
                                            dmc.Text(
                                                "Feature Type",
                                                size="sm",
                                                fw=500,  # pyright: ignore[reportArgumentType]
                                                opacity=0.8,  # pyright: ignore[reportArgumentType]
                                            ),
                                            dmc.Switch(
                                                id=aio_id(
                                                    "FeatureSpace",
                                                    self.aio_id,
                                                    "feature-type-switch",
                                                ),
                                                label="Filtered",
                                                checked=use_filtered,
                                                size="md",
                                            ),
                                        ],
                                        gap="xs",
                                        style={
                                            "display": "block" if show_feature_switch else "none"
                                        },
                                    ),
                                ],
                                direction={"base": "row", "md": "column"},  # pyright: ignore[reportArgumentType]
                                gap="md",
                                wrap="wrap",
                                style={
                                    "padding": "16px 8px 16px 16px",
                                    "@media (min-width: 992px)": {"padding": "16px"},
                                },
                            ),
                        ],
                        span={"base": 12, "md": 3},  # pyright: ignore[reportArgumentType]
                        style={"borderRight": "1px solid #373A40"},
                    ),
                    dmc.GridCol(
                        [
                            dcc.Graph(
                                id=aio_id("FeatureSpace", self.aio_id, "plot"),
                                figure=self.build_figure(
                                    x_axis=x_axis,
                                    y_axis=y_axis,
                                    selected_labels=selected_labels,
                                    use_filtered=use_filtered,
                                ),
                                style={
                                    "width": "100%",
                                    "height": "calc(100vh - 60px)",
                                },
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                },
                            ),
                        ],
                        span={"base": 12, "md": 9},  # pyright: ignore[reportArgumentType]
                    ),
                ],
            ),
            self.trajectory_modal.render(),
        ])

    def build_figure(
        self,
        x_axis: int = 0,
        y_axis: int | None = 1,
        selected_labels: list[str] | None = None,
        use_filtered: bool = True,
    ) -> go.Figure:
        """Build feature space scatter plot."""
        fig = go.Figure()

        if self.bse.solution is None:
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="No data available. Run Basin Stability Estimation first.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        logger.debug("Building figure: use_filtered=%s", use_filtered)
        logger.debug("features available: %s", self.bse.solution.features is not None)
        logger.debug(
            "extracted_features available: %s",
            self.bse.solution.extracted_features is not None,
        )
        if self.bse.solution.features is not None:
            logger.debug("features shape: %s", self.bse.solution.features.shape)
        if self.bse.solution.extracted_features is not None:
            logger.debug("extracted_features shape: %s", self.bse.solution.extracted_features.shape)

        if use_filtered and self.bse.solution.features is not None:
            features = self.bse.solution.features.cpu().numpy()
        elif not use_filtered and self.bse.solution.extracted_features is not None:
            features = self.bse.solution.extracted_features.cpu().numpy()
        else:
            error_msg = "No features available. "
            if use_filtered:
                error_msg += "Filtered features not found."
            else:
                error_msg += "Extracted features not found."
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text=error_msg,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        labels = None
        if self.bse.solution.labels is not None and len(self.bse.solution.labels) > 0:
            labels = np.array(self.bse.solution.labels)

        if labels is None:
            return fig

        # Filter out unbounded trajectories (features are only computed for bounded ones)
        bounded_mask = labels != "unbounded"
        if bounded_mask.sum() == 0:
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="All trajectories are unbounded. No features to display.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        # Check if features array matches bounded trajectory count
        n_bounded = int(bounded_mask.sum())
        if features.shape[0] != n_bounded:
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text=f"Feature array size mismatch: {features.shape[0]} features vs {n_bounded} bounded trajectories",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        # Filter labels to only bounded trajectories (features are already filtered)
        # Keep track of original indices for click handling
        original_indices = np.where(bounded_mask)[0]
        labels = labels[bounded_mask]

        unique_labels = np.unique(labels)

        if selected_labels is not None:
            unique_labels = np.array([
                label for label in unique_labels if str(label) in selected_labels
            ])

        feature_options = self.get_feature_options(use_filtered=use_filtered)
        n_features = features.shape[1]

        if x_axis >= n_features:
            x_axis = 0
        if y_axis is None or y_axis >= n_features:
            y_axis = 0

        x_label = (
            feature_options[x_axis]["label"]
            if x_axis < len(feature_options)
            else f"Feature {x_axis}"
        )

        y_label = (
            feature_options[y_axis]["label"]
            if y_axis < len(feature_options)
            else f"Feature {y_axis}"
        )

        if n_features == 1:
            rng = np.random.default_rng(42)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                indices = original_indices[mask]
                x_data = features[mask, x_axis]
                y_data = rng.uniform(-0.4, 0.4, size=len(x_data))

                # customdata must be 2D array - each point gets a list of values
                customdata_2d = [[idx] for idx in indices]

                scatter_constructor = go.Scattergl if len(x_data) > 5000 else go.Scatter
                fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                    scatter_constructor(
                        x=x_data,
                        y=y_data,
                        mode="markers",
                        name=str(label),
                        customdata=customdata_2d,
                        marker={
                            "size": 8,
                            "color": get_color(i),
                            "opacity": 0.6,
                        },
                        hovertemplate=(
                            f"<b>{label}</b><br>" + f"{x_label}: %{{x:.4f}}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
                title="Feature Space (1D Strip Plot)",
                xaxis_title=x_label,
                yaxis_title="",
                yaxis={"visible": False},
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.0,
                    "xanchor": "left",
                    "x": 0,
                },
                hovermode="closest",
            )
        else:
            for i, label in enumerate(unique_labels):
                mask = labels == label
                indices = original_indices[mask]
                x_data = features[mask, x_axis]
                y_data = features[mask, y_axis]

                # customdata must be 2D array - each point gets a list of values
                customdata_2d = [[idx] for idx in indices]

                scatter_constructor = go.Scattergl if len(x_data) > 5000 else go.Scatter
                fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                    scatter_constructor(
                        x=x_data,
                        y=y_data,
                        mode="markers",
                        name=str(label),
                        customdata=customdata_2d,
                        marker={
                            "size": 4,
                            "color": get_color(i),
                            "opacity": 0.7,
                        },
                        hovertemplate=(
                            f"<b>{label}</b><br>"
                            + f"{x_label}: %{{x:.4f}}<br>"
                            + f"{y_label}: %{{y:.4f}}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
                title="Feature Space",
                xaxis_title=x_label,
                yaxis_title=y_label,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "left",
                    "x": 0,
                },
                hovermode="closest",
            )

        return fig


@callback(
    [
        Output(aio_id("FeatureSpace", MATCH, "plot"), "figure"),
        Output(aio_id("FeatureSpace", MATCH, "x-select"), "data"),
        Output(aio_id("FeatureSpace", MATCH, "y-select"), "data"),
        Output(aio_id("FeatureSpace", MATCH, "x-select"), "value"),
        Output(aio_id("FeatureSpace", MATCH, "y-select"), "value"),
    ],
    [
        Input(aio_id("FeatureSpace", MATCH, "x-select"), "value"),
        Input(aio_id("FeatureSpace", MATCH, "y-select"), "value"),
        Input(aio_id("FeatureSpace", MATCH, "label-select"), "value"),
        Input(aio_id("FeatureSpace", MATCH, "feature-type-switch"), "checked"),
    ],
    State(aio_id("FeatureSpace", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_feature_space_figure_aio(
    x_axis_str: str,
    y_axis_str: str | None,
    selected_labels: list[str],
    use_filtered: bool,
    plot_id: dict[str, Any],
) -> tuple[go.Figure, list[dict[str, str]], list[dict[str, str]], str, str | None]:
    """Update figure and dropdowns when controls change."""
    instance_id = plot_id["aio_id"]
    instance = FeatureSpaceAIO.get_instance(instance_id)
    if instance is None or not isinstance(instance, FeatureSpaceAIO):
        return go.Figure(), [], [], "0", "1"

    # Check if feature type switch was toggled
    triggered_id = cast(str | dict[str, Any] | None, ctx.triggered_id)  # pyright: ignore[reportUnknownMemberType]
    feature_switch_toggled = False
    if isinstance(triggered_id, dict):
        feature_switch_toggled = triggered_id.get("subcomponent") == "feature-type-switch"

    # Get new feature options based on use_filtered
    feature_options = instance.get_feature_options(use_filtered=use_filtered)

    if feature_switch_toggled:
        # Reset to first two features when switching
        new_x = "0"
        new_y = "1" if len(feature_options) > 1 else None
        x_axis = 0
        y_axis = 1 if len(feature_options) > 1 else None
    else:
        # Keep current selections
        new_x = x_axis_str
        new_y = y_axis_str
        x_axis = int(x_axis_str)
        y_axis = int(y_axis_str) if y_axis_str is not None else None

    figure = instance.build_figure(
        x_axis=x_axis,
        y_axis=y_axis,
        selected_labels=selected_labels,
        use_filtered=use_filtered,
    )

    return figure, feature_options, feature_options, new_x, new_y


@callback(
    [
        Output(aio_id("TrajectoryModal", MATCH, "modal"), "opened", allow_duplicate=True),
        Output(aio_id("TrajectoryModal", MATCH, "modal"), "title", allow_duplicate=True),
        Output(aio_id("TrajectoryModal", MATCH, "info"), "children", allow_duplicate=True),
        Output(aio_id("TrajectoryModal", MATCH, "sample-store"), "data", allow_duplicate=True),
    ],
    Input(aio_id("FeatureSpace", MATCH, "plot"), "clickData"),
    State(aio_id("FeatureSpace", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def open_trajectory_modal_from_feature_space_aio(
    click_data: dict[str, Any] | None,
    plot_id: dict[str, Any],
) -> tuple[
    bool | NoUpdate,
    str | NoUpdate,
    str | NoUpdate,
    dict[str, Any] | None | NoUpdate,
]:
    """Open trajectory modal when a point is clicked."""
    if click_data is None:
        return no_update, no_update, no_update, no_update

    instance_id = plot_id["aio_id"]
    instance = FeatureSpaceAIO.get_instance(instance_id)
    if instance is None:
        return no_update, no_update, no_update, no_update

    try:
        point = click_data["points"][0]
        customdata = point.get("customdata")

        # customdata is a list like [sample_idx], extract the first element
        if customdata is None or not isinstance(customdata, list) or len(customdata) == 0:  # pyright: ignore[reportUnknownArgumentType]
            return no_update, no_update, no_update, no_update

        sample_idx = int(customdata[0])  # type: ignore[arg-type]

        x_value: float = point.get("x", 0.0)
        y_value: float = point.get("y", 0.0)
        x_label: str = (
            click_data.get("points", [{}])[0].get("xaxis", {}).get("title", {}).get("text", "X")
        )
        y_label: str = (
            click_data.get("points", [{}])[0].get("yaxis", {}).get("title", {}).get("text", "Y")
        )

        title = f"Trajectory {sample_idx}"
        info = f"Clicked on {x_label} = {x_value:.4f}, {y_label} = {y_value:.4f}"

        sample_data: dict[str, Any] = {
            "sample_idx": sample_idx,
            "x_label": x_label,
            "y_label": y_label,
            "x_value": x_value,
            "y_value": y_value,
        }

        return True, title, info, sample_data

    except (KeyError, IndexError, TypeError):
        return no_update, no_update, no_update, no_update
