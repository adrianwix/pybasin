"""AIO Template Time Series page."""

from collections.abc import Sequence
from typing import Any, cast

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from dash import (
    MATCH,
    Input,
    Output,
    State,
    callback,  # pyright: ignore[reportUnknownVariableType]
    dcc,
    html,
)
from dash.development.base_component import Component
from plotly.subplots import (  # pyright: ignore[reportMissingTypeStubs]
    make_subplots,  # pyright: ignore[reportUnknownVariableType]
)

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.trajectory_cache import TrajectoryCache
from pybasin.plotters.interactive_plotter.utils import get_color
from pybasin.plotters.types import TemplatesTimeSeriesOptions, filter_by_include_exclude


class TemplateTimeSeriesAIO(BseBasePageAIO):
    """
    AIO component for template trajectory time series plots.

    Displays time series for template trajectories.
    """

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
        options: TemplatesTimeSeriesOptions | None = None,
    ):
        """
        Initialize template time series AIO component.

        :param bse: Basin stability estimator with computed results.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to labels.
        :param options: Template time series configuration options.
        """
        super().__init__(bse, aio_id, state_labels)
        self.options = options or {}
        TemplateTimeSeriesAIO._instances[aio_id] = self

    def get_time_bounds(self) -> tuple[float, float]:
        """Get time bounds from integrated solution."""
        t, _ = TrajectoryCache.get_or_integrate(self.bse)
        return float(t[0]), float(t[-1])

    def render(self) -> html.Div:
        """Render complete page layout with controls and time series plot."""
        if self.bse.template_integrator is None:
            return html.Div(
                dmc.Paper(
                    dmc.Text(
                        "Template time series requires a template_integrator",
                        size="lg",
                        c="gray",
                    ),
                    p="xl",
                    withBorder=True,
                )
            )

        state_options = self.get_state_options()
        all_template_labels = self.bse.template_integrator.labels
        selected_templates = filter_by_include_exclude(
            all_template_labels,
            self.options.get("include_templates"),
            self.options.get("exclude_templates"),
        )

        t_min, t_max = self.get_time_bounds()
        time_range = self.options.get("time_range", (0.85, 1.0))
        default_span = (
            t_min + (t_max - t_min) * time_range[0],
            t_min + (t_max - t_min) * time_range[1],
        )

        state_variable = self.options.get("state_variable", 0)
        select_data = cast(Sequence[str], state_options)

        controls: list[Component] = [
            dmc.Select(
                id=aio_id("TemplateTimeSeries", self.aio_id, "state-select"),
                label="State Variable",
                data=select_data,
                value=str(state_variable),
                w=150,
            ),
            dmc.MultiSelect(
                id=aio_id("TemplateTimeSeries", self.aio_id, "template-select"),
                label="Templates",
                data=all_template_labels,
                value=selected_templates,
                w=300,
            ),
        ]

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.Group(controls, gap="md"),
                        dmc.Space(h="md"),
                        dmc.Text("Time Range", size="sm", fw="normal"),
                        dcc.RangeSlider(
                            id=aio_id("TemplateTimeSeries", self.aio_id, "time-slider"),
                            min=t_min,
                            max=t_max,
                            step=(t_max - t_min) / 100,
                            value=[default_span[0], default_span[1]],
                            marks={
                                t_min: {"label": f"{t_min:.1f}"},
                                t_max: {"label": f"{t_max:.1f}"},
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    p="md",
                    withBorder=True,
                    mb="md",
                ),
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=aio_id("TemplateTimeSeries", self.aio_id, "plot"),
                            figure=self.build_figure(
                                state_variable=state_variable,
                                time_span=default_span,
                                selected_templates=selected_templates,
                            ),
                            style={"height": "70vh"},
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                            },
                        ),
                    ],
                    p="md",
                    withBorder=True,
                ),
            ],
            style={"padding": "16px"},
        )

    def build_figure(
        self,
        state_variable: int = 0,
        time_span: tuple[float, float] | None = None,
        selected_templates: list[str] | None = None,
    ) -> go.Figure:
        """Build stacked template time series plot with one subplot per trajectory."""
        if self.bse.template_integrator is None:
            fig = go.Figure()
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="Template plot requires a template_integrator",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        t, y = TrajectoryCache.get_or_integrate(self.bse)

        if time_span is not None:
            mask = (t >= time_span[0]) & (t <= time_span[1])
            t = t[mask]
            y = y[:, :, mask]

        all_labels = self.bse.template_integrator.labels
        if selected_templates is None:
            selected_templates = all_labels

        filtered_indices: list[int] = [
            i for i, label in enumerate(all_labels) if label in selected_templates
        ]
        n_plots = len(filtered_indices)

        if n_plots == 0:
            fig = go.Figure()
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="No templates selected",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        y_limits = self.options.get("y_limits")

        row_titles = [f"ȳ<sub>{i + 1}</sub>" for i in filtered_indices]
        fig = make_subplots(
            rows=n_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_titles=row_titles,
        )

        trajectories = np.transpose(y, (0, 2, 1))  # (n_templates, n_time, n_states)

        for row_idx, i in enumerate(filtered_indices, start=1):
            label = all_labels[i]
            traj = trajectories[i]
            t_plot = t
            y_plot = traj[:, state_variable]

            fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=t_plot,
                    y=y_plot,
                    mode="lines",
                    name=label,
                    line={"color": get_color(i), "width": 1.5},
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

            y_lim = y_limits[label] if isinstance(y_limits, dict) else y_limits
            if y_lim is not None:
                y_min, y_max = y_lim
                fig.update_yaxes(  # pyright: ignore[reportUnknownMemberType]
                    range=[y_min, y_max],
                    row=row_idx,
                    col=1,
                )

        fig.update_xaxes(  # pyright: ignore[reportUnknownMemberType]
            title_text="time",
            row=n_plots,
            col=1,
        )
        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title=f"Template Trajectories - {self.get_state_label(state_variable)}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=max(400, 120 * n_plots),
        )

        return fig


@callback(
    Output(aio_id("TemplateTimeSeries", MATCH, "plot"), "figure"),
    [
        Input(aio_id("TemplateTimeSeries", MATCH, "state-select"), "value"),
        Input(aio_id("TemplateTimeSeries", MATCH, "template-select"), "value"),
        Input(aio_id("TemplateTimeSeries", MATCH, "time-slider"), "value"),
    ],
    State(aio_id("TemplateTimeSeries", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_template_time_series_figure_aio(
    state_variable: str,
    templates: list[str],
    time_range: list[float],
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update time series plot when controls change."""
    instance_id = plot_id["aio_id"]
    instance = TemplateTimeSeriesAIO.get_instance(instance_id)
    if instance is None or not isinstance(instance, TemplateTimeSeriesAIO):
        return go.Figure()

    return instance.build_figure(
        state_variable=int(state_variable),
        time_span=(time_range[0], time_range[1]),
        selected_templates=templates,
    )
