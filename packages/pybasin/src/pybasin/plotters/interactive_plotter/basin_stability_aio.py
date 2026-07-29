"""AIO Basin Stability bar chart page."""

from typing import cast

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from dash import dcc, html

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.utils import get_color, tensor_to_float_list


class BasinStabilityAIO(BseBasePageAIO):
    """
    AIO component for basin stability bar chart.

    Displays bar chart with basin stability values and info panel
    with ODE system details. No interactive controls needed.
    """

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize basin stability AIO component.

        :param bse: Basin stability estimator with computed results.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to labels.
        """
        super().__init__(bse, aio_id, state_labels)

    def get_sampler_name(self) -> str:
        """Get display name of the sampler."""
        sampler = self.bse.sampler
        display_name = getattr(sampler, "display_name", None)
        return display_name or sampler.__class__.__name__

    def get_solver_name(self) -> str:
        """Get display name of the solver."""
        solver = self.bse.solver
        display_name = getattr(solver, "display_name", None)
        return display_name or solver.__class__.__name__

    def get_classifier_name(self) -> str:
        """Get display name of the cluster classifier."""
        classifier = self.bse.predictor
        display_name = getattr(classifier, "display_name", None)
        return display_name or classifier.__class__.__name__

    def get_feature_extractor_name(self) -> str:
        """Get display name of the feature extractor."""
        fe = self.bse.feature_extractor
        display_name = getattr(fe, "display_name", None)
        return display_name or fe.__class__.__name__

    def render(self) -> html.Div:
        """Render complete page layout with info panel and bar chart."""
        return html.Div(
            [
                dmc.Grid(
                    [
                        dmc.GridCol(
                            self._build_info_panel(),
                            span=4,
                        ),
                        dmc.GridCol(
                            dmc.Paper(
                                [
                                    dcc.Graph(
                                        id=aio_id("BasinStability", self.aio_id, "plot"),
                                        figure=self.build_figure(),
                                        style={"height": "60vh"},
                                        config={
                                            "displayModeBar": True,
                                            "scrollZoom": True,
                                        },
                                    ),
                                ],
                                p="md",
                                withBorder=True,
                            ),
                            span="auto",
                        ),
                    ],
                    gutter="md",
                ),
            ],
            style={"padding": "16px"},
        )

    def _build_info_panel(self) -> dmc.Paper:
        """Build information panel showing ODE system and sampler details."""
        ode_str = self.bse.ode_system.get_str()

        sampler = self.bse.sampler
        min_limits = tensor_to_float_list(sampler.min_limits)
        max_limits = tensor_to_float_list(sampler.max_limits)

        state_ranges: list[dmc.Text] = []
        for i, (min_val, max_val) in enumerate(zip(min_limits, max_limits, strict=True)):
            label = self.get_state_label(i)
            state_ranges.append(
                dmc.Text(
                    f"{label}: [{min_val:.4g}, {max_val:.4g}]",
                    size="sm",
                    ff="monospace",
                )
            )

        ode_params_section: list[dmc.Text | dmc.Stack] = []
        if hasattr(self.bse.ode_system, "params"):
            params = self.bse.ode_system.params
            if isinstance(params, dict):
                params_dict = cast(dict[str, float], params)
                param_items: list[dmc.Group] = []
                for key, value in params_dict.items():
                    formatted_value = f"{value:.4g}" if isinstance(value, float) else str(value)
                    param_items.append(
                        dmc.Group(
                            [
                                dmc.Text(f"{key}:", size="sm", fw="normal", ff="monospace"),
                                dmc.Text(formatted_value, size="sm", ff="monospace"),
                            ],
                            gap="xs",
                        )
                    )
                ode_params_section = [
                    dmc.Text("Parameters", fw="bold", size="lg", mt="sm"),
                    dmc.Stack(param_items, gap="xs", ml="md"),
                ]

        return dmc.Paper(
            [
                dmc.Stack(
                    [
                        dmc.Text("ODE System", fw="bold", size="lg"),
                        dmc.Code(
                            ode_str,
                            block=True,
                        ),
                        *ode_params_section,
                        dmc.Divider(my="sm"),
                        dmc.Text("Sampler Configuration", fw="bold", size="lg"),
                        dmc.Group(
                            [
                                dmc.Text("Type:", size="sm", fw="normal"),
                                dmc.Badge(
                                    self.get_sampler_name(),
                                    color="blue",
                                    variant="light",
                                    style={"textTransform": "none"},
                                ),
                            ],
                            gap="xs",
                        ),
                        dmc.Text("Sample Limits", fw="normal", size="sm", mt="xs"),
                        dmc.Stack(state_ranges, gap="xs", ml="md"),
                        dmc.Divider(my="sm"),
                        dmc.Text("Solver", fw="bold", size="lg"),
                        dmc.Badge(
                            self.get_solver_name(),
                            color="green",
                            variant="light",
                            style={"textTransform": "none"},
                        ),
                        dmc.Divider(my="sm"),
                        dmc.Text("Feature Extractor", fw="bold", size="lg"),
                        dmc.Badge(
                            self.get_feature_extractor_name(),
                            color="orange",
                            variant="light",
                            style={"textTransform": "none"},
                        ),
                        dmc.Divider(my="sm"),
                        dmc.Text("Cluster Classifier", fw="bold", size="lg"),
                        dmc.Badge(
                            self.get_classifier_name(),
                            color="violet",
                            variant="light",
                            style={"textTransform": "none"},
                        ),
                    ],
                    gap="xs",
                ),
            ],
            p="md",
            withBorder=True,
        )

    def build_figure(self) -> go.Figure:
        """Build basin stability bar chart."""
        fig = go.Figure()

        if self.bse.result is None:
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

        labels = list(self.bse.result["basin_stability"].keys())
        values = list(self.bse.result["basin_stability"].values())
        colors = [get_color(i) for i in range(len(labels))]

        fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
            go.Bar(
                x=labels,
                y=values,
                marker={"color": colors},
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Basin Stability: %{y:.4f}<extra></extra>",
            )
        )

        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title="Basin Stability",
            xaxis_title="Attractor",
            yaxis_title="Basin Stability",
            yaxis={"range": [0, 1.05]},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )

        return fig
