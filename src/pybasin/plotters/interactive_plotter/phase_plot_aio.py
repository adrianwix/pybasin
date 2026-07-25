"""AIO Phase Plot page for 2D and 3D template trajectories."""

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

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.trajectory_cache import TrajectoryCache
from pybasin.plotters.interactive_plotter.utils import get_color
from pybasin.plotters.types import (
    TemplatesPhaseSpaceOptions,
    filter_by_include_exclude,
    infer_z_axis,
)


class PhasePlotAIO(BseBasePageAIO):
    """
    AIO component for 2D or 3D phase plot of template trajectories.

    Supports dynamic axis selection and template filtering.
    """

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        aio_id: str,
        is_3d: bool = False,
        state_labels: dict[int, str] | None = None,
        options: TemplatesPhaseSpaceOptions | None = None,
    ):
        """
        Initialize phase plot AIO component.

        :param bse: Basin stability estimator with computed results.
        :param aio_id: Unique identifier for this component instance.
        :param is_3d: Deprecated - kept for compatibility. Plot is 2D/3D based on Z selection.
        :param state_labels: Optional mapping of state indices to labels.
        :param options: Phase plot configuration options.
        """
        super().__init__(bse, aio_id, state_labels)
        self.is_3d = is_3d
        self.options = options or {}
        PhasePlotAIO._instances[aio_id] = self

    def _get_template_labels(self) -> list[str]:
        """Get list of template labels if template_integrator is available."""
        if self.bse.template_integrator is not None:
            return list(self.bse.template_integrator.labels)
        return []

    def render(self) -> html.Div:
        """Render complete page layout with controls and phase plot."""
        if self.bse.template_integrator is None:
            return html.Div(
                dmc.Paper(
                    dmc.Text(
                        "Phase plot requires a template_integrator with template ICs",
                        size="lg",
                        c="gray",
                    ),
                    p="xl",
                    withBorder=True,
                )
            )

        n_states = self.get_n_states()
        state_options = self.get_state_options()
        all_template_labels = self._get_template_labels()
        selected_templates = filter_by_include_exclude(
            all_template_labels,
            self.options.get("include_templates"),
            self.options.get("exclude_templates"),
        )

        select_data = cast(Sequence[str], state_options)
        default_x = str(self.options.get("x_axis", 0))
        default_y = str(self.options.get("y_axis", 1)) if n_states > 1 else "0"

        controls: list[dmc.Select | dmc.MultiSelect] = [
            dmc.Select(
                id=aio_id("PhasePlot", self.aio_id, "x-select"),
                label="X Axis",
                data=select_data,
                value=default_x,
                w=150,
            ),
            dmc.Select(
                id=aio_id("PhasePlot", self.aio_id, "y-select"),
                label="Y Axis",
                data=select_data,
                value=default_y,
                w=150,
            ),
        ]

        z_var = (
            infer_z_axis(
                self.options.get("x_axis", 0),
                self.options.get("y_axis", 1),
                n_states,
            )
            if n_states > 2
            else None
        )
        default_z = str(z_var) if z_var is not None else None
        controls.append(
            dmc.Select(
                id=aio_id("PhasePlot", self.aio_id, "z-select"),
                label="Z Axis",
                data=select_data,
                value=default_z,
                w=200,
                clearable=True,
                disabled=n_states < 3,
            )
        )

        controls.append(
            dmc.MultiSelect(
                id=aio_id("PhasePlot", self.aio_id, "template-select"),
                label="Templates",
                data=all_template_labels if all_template_labels else [],
                value=selected_templates if all_template_labels else [],
                w=300,
                disabled=not all_template_labels,
            )
        )

        z_var = (
            infer_z_axis(
                self.options.get("x_axis", 0),
                self.options.get("y_axis", 1),
                n_states,
            )
            if n_states > 2
            else None
        )

        return html.Div([
            dmc.Paper(
                dmc.Group(controls, gap="md"),
                p="md",
                withBorder=True,
                mb="md",
            ),
            dmc.Paper(
                [
                    dcc.Graph(
                        id=aio_id("PhasePlot", self.aio_id, "plot"),
                        figure=self.build_figure(
                            x_var=int(default_x),
                            y_var=int(default_y),
                            z_var=z_var,
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
        ])

    def build_figure(
        self,
        x_var: int = 0,
        y_var: int = 1,
        z_var: int | None = None,
        selected_templates: list[str] | None = None,
    ) -> go.Figure:
        """Build phase plot figure."""
        if self.bse.template_integrator is None:
            fig = go.Figure()
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="Phase plot requires a template_integrator with template ICs",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        _t, trajectories = TrajectoryCache.get_or_integrate(self.bse)

        fig = go.Figure()

        all_labels = self.bse.template_integrator.labels
        if selected_templates is None:
            selected_templates = list(all_labels)

        for i, (label, traj) in enumerate(
            zip(
                all_labels,
                np.transpose(trajectories, (0, 2, 1)),
                strict=True,
            )
        ):
            if label not in selected_templates:
                continue

            if z_var is not None:
                fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                    go.Scatter3d(
                        x=traj[:, x_var],
                        y=traj[:, y_var],
                        z=traj[:, z_var],
                        mode="lines",
                        name=str(label),
                        line={"color": get_color(i), "width": 3},
                    )
                )
            else:
                fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                    go.Scatter(
                        x=traj[:, x_var],
                        y=traj[:, y_var],
                        mode="lines",
                        name=str(label),
                        line={"color": get_color(i), "width": 2},
                    )
                )

        title = "Phase Plot 3D" if z_var is not None else "Phase Plot 2D"
        x_label = self.get_state_label(x_var)
        y_label = self.get_state_label(y_var)

        if z_var is not None:
            z_label = self.get_state_label(z_var)
            fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
                title=title,
                scene={
                    "xaxis_title": x_label,
                    "yaxis_title": y_label,
                    "zaxis_title": z_label,
                },
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
            )
        else:
            fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

        return fig


@callback(
    Output(aio_id("PhasePlot", MATCH, "plot"), "figure"),
    [
        Input(aio_id("PhasePlot", MATCH, "x-select"), "value"),
        Input(aio_id("PhasePlot", MATCH, "y-select"), "value"),
        Input(aio_id("PhasePlot", MATCH, "z-select"), "value"),
        Input(aio_id("PhasePlot", MATCH, "template-select"), "value"),
    ],
    State(aio_id("PhasePlot", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_phase_plot_figure_aio(
    x_var: str,
    y_var: str,
    z_var: str | None,
    templates: list[str],
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update phase plot when controls change."""
    instance_id = plot_id["aio_id"]
    instance = PhasePlotAIO.get_instance(instance_id)
    if instance is None or not isinstance(instance, PhasePlotAIO):
        return go.Figure()

    z_var_int = None
    if z_var is not None and z_var != "":
        try:
            z_var_int = int(z_var)
        except (ValueError, TypeError):
            z_var_int = None

    return instance.build_figure(
        x_var=int(x_var),
        y_var=int(y_var),
        z_var=z_var_int,
        selected_templates=templates if templates else None,
    )
