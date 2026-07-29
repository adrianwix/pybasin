"""AIO Parameter Overview page showing basin stability across parameter values."""

from collections import defaultdict
from typing import Any

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
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

from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.utils import get_color


class ParamOverviewAIO:
    """
    AIO component for parameter overview page.

    Shows basin stability evolution across parameter sweep values.
    Supports multi-parameter studies by grouping results along one chosen parameter
    (x-axis) while producing separate curves for each combination of the remaining
    parameters.
    """

    _instances: dict[str, "ParamOverviewAIO"] = {}

    @classmethod
    def get_instance(cls, instance_id: str) -> "ParamOverviewAIO | None":
        """Get instance by ID."""
        return cls._instances.get(instance_id)

    def __init__(
        self,
        bs_study: BasinStabilityStudy,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize parameter overview AIO component.

        :param bs_study: Basin stability study instance.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to labels.
        """
        self.bs_study = bs_study
        self.aio_id = aio_id
        self.state_labels = state_labels or {}
        ParamOverviewAIO._instances[aio_id] = self

    def get_parameter_names(self) -> list[str]:
        """Get all studied parameter names."""
        return self.bs_study.studied_parameter_names

    def get_parameter_options(self) -> list[dict[str, str]]:
        """Get dropdown options for parameter selection."""
        return [{"value": name, "label": name} for name in self.get_parameter_names()]

    def _get_all_labels(self) -> list[str]:
        """Get all unique labels across all parameter values."""
        all_labels: set[str] = set()
        for r in self.bs_study.results:
            all_labels.update(r["basin_stability"].keys())
        return sorted(all_labels)

    def _group_by_parameter(self, param_name: str) -> dict[tuple[tuple[str, Any], ...], list[int]]:
        """Group study result indices by the values of all parameters except param_name.

        Within each group the indices are sorted by param_name's value so they
        can be plotted as a line.

        :param param_name: The parameter whose values form the x-axis.
        :return: Mapping from a tuple of (other_param, value) pairs to sorted result indices.
        """
        other_params = [p for p in self.get_parameter_names() if p != param_name]

        groups: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
        for i, r in enumerate(self.bs_study.results):
            sl = r["study_label"]
            group_key = tuple((p, sl[p]) for p in other_params) if other_params else ()
            groups[group_key].append(i)

        for group_key in groups:
            groups[group_key].sort(
                key=lambda i: self.bs_study.results[i]["study_label"][param_name]
            )

        return dict(groups)

    def render(self) -> html.Div:
        """Render complete page layout with controls and plot."""
        all_labels = self._get_all_labels()
        label_options = [{"value": label, "label": label} for label in all_labels]
        param_options = self.get_parameter_options()
        default_param = param_options[0]["value"] if param_options else ""

        return html.Div([
            dmc.Paper(
                [
                    dmc.Group(
                        [
                            dmc.Select(
                                id=aio_id("ParamOverview", self.aio_id, "x_param"),
                                label="X-Axis Parameter",
                                data=param_options,  # type: ignore[arg-type]
                                value=default_param,
                                w=200,
                            ),
                            dmc.SegmentedControl(
                                id=aio_id("ParamOverview", self.aio_id, "scale"),
                                data=[  # pyright: ignore[reportArgumentType]
                                    {"value": "linear", "label": "Linear"},
                                    {"value": "log", "label": "Log"},
                                ],
                                value="linear",
                            ),
                            dmc.MultiSelect(
                                id=aio_id("ParamOverview", self.aio_id, "labels"),
                                label="Show Labels",
                                # Typing is bad: https://www.dash-mantine-components.com/components/multiselect
                                data=label_options,  # pyright: ignore[reportArgumentType]
                                value=all_labels,
                                w=300,
                            ),
                        ],
                        gap="md",
                    ),
                ],
                p="md",
                withBorder=True,
            ),
            dmc.Paper(
                [
                    dcc.Graph(
                        id=aio_id("ParamOverview", self.aio_id, "plot"),
                        figure=self.build_figure(x_param=default_param),
                        style={"height": "calc(100vh - 190px)"},
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
        x_param: str | None = None,
        x_scale: str = "linear",
        label_filter: list[str] | None = None,
    ) -> go.Figure:
        """Build basin stability variation figure.

        :param x_param: Parameter to use as x-axis. Defaults to first parameter.
        :param x_scale: Scale for x-axis ('linear' or 'log').
        :param label_filter: List of attractor labels to show. None shows all.
        :return: Plotly figure.
        """
        params = self.get_parameter_names()
        if not params:
            return go.Figure()

        if x_param is None or x_param not in params:
            x_param = params[0]

        all_labels = self._get_all_labels()
        labels_to_show = label_filter if label_filter else all_labels

        groups = self._group_by_parameter(x_param)
        n_groups = len(groups)

        markers = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]
        linestyles = ["solid", "dash", "dot", "dashdot"]

        fig = go.Figure()

        for g_idx, (group_key, indices) in enumerate(groups.items()):
            x_values = [self.bs_study.results[i]["study_label"][x_param] for i in indices]

            group_suffix = ""
            if group_key:
                group_suffix = " (" + ", ".join(f"{k}={v}" for k, v in group_key) + ")"

            for a_idx, label in enumerate(all_labels):
                if label not in labels_to_show:
                    continue

                y_values = [
                    self.bs_study.results[i]["basin_stability"].get(label, 0) for i in indices
                ]

                # Use group color, attractor marker/linestyle
                fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines+markers",
                        name=f"{label}{group_suffix}",
                        line={
                            "color": get_color(g_idx) if n_groups > 1 else get_color(a_idx),
                            "dash": linestyles[a_idx % len(linestyles)],
                        },
                        marker={
                            "size": 8,
                            "symbol": markers[a_idx % len(markers)],
                        },
                        legendgroup=f"group_{g_idx}" if n_groups > 1 else f"label_{a_idx}",
                    )
                )

        fig.update_xaxes(  # pyright: ignore[reportUnknownMemberType]
            type="log" if x_scale == "log" else "linear",
            title=x_param,
            autorange=True,
        )
        fig.update_yaxes(title="Basin Stability", range=[0, 1], autorange=True)  # pyright: ignore[reportUnknownMemberType]
        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title=f"Basin Stability vs {x_param}",
            hovermode="x unified",
            template="plotly_dark",
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
            },
        )

        return fig


@callback(
    Output(aio_id("ParamOverview", MATCH, "plot"), "figure"),
    [
        Input(aio_id("ParamOverview", MATCH, "x_param"), "value"),
        Input(aio_id("ParamOverview", MATCH, "scale"), "value"),
        Input(aio_id("ParamOverview", MATCH, "labels"), "value"),
    ],
    State(aio_id("ParamOverview", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_param_overview_figure_aio(
    x_param: str,
    x_scale: str,
    label_filter: list[str],
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update parameter overview figure when controls change."""
    instance_id = plot_id["aio_id"]
    instance = ParamOverviewAIO.get_instance(instance_id)
    if instance is None:
        return go.Figure()

    return instance.build_figure(x_param=x_param, x_scale=x_scale, label_filter=label_filter)
