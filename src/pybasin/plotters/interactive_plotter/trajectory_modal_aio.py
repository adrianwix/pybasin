"""AIO Trajectory Modal component with instance-scoped state."""

from typing import Any

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from dash import Input, Output, State, callback, dcc  # pyright: ignore[reportUnknownVariableType]

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.utils import get_color


class TrajectoryModalAIO:
    """
    AIO component for trajectory inspection modal.

    Each instance maintains its own state using pattern-matching IDs,
    eliminating the need for global SELECTED_SAMPLE_DATA store.
    """

    _instances: dict[str, "TrajectoryModalAIO"] = {}

    @classmethod
    def get_instance(cls, instance_id: str) -> "TrajectoryModalAIO | None":
        """Get instance by ID."""
        return cls._instances.get(instance_id)

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        instance_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize trajectory modal AIO component.

        :param bse: Basin stability estimator with computed results.
        :param instance_id: Unique identifier for this modal instance.
        :param state_labels: Optional mapping of state indices to labels.
        """
        self.bse = bse
        self.instance_id = instance_id
        self.state_labels = state_labels or {}
        TrajectoryModalAIO._instances[instance_id] = self

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def get_n_states(self) -> int:
        """Get number of state variables."""
        if self.bse.y0 is None:
            return 0
        return self.bse.y0.shape[1]

    def get_time_bounds(self) -> tuple[float, float]:
        """Get time bounds from solution."""
        if self.bse.solution is None:
            return 0.0, 1.0
        t = self.bse.solution.time.cpu().numpy()
        return float(t[0]), float(t[-1])

    def render(self, default_time_range_percent: float = 1.0) -> dmc.Modal:
        """
        Render the modal layout with pattern-matching IDs.

        :param default_time_range_percent: Default time range as percentage of total.
        :return: Modal component with instance-scoped IDs.
        """
        n_states = self.get_n_states()
        state_options = [
            {"value": str(i), "label": self.get_state_label(i)} for i in range(n_states)
        ]

        t_min, t_max = self.get_time_bounds()
        default_t_max = t_min + (t_max - t_min) * default_time_range_percent

        return dmc.Modal(
            id=aio_id("TrajectoryModal", self.instance_id, "modal"),
            title="Trajectory Viewer",
            size="xl",
            centered=True,
            children=[
                dcc.Store(
                    id=aio_id("TrajectoryModal", self.instance_id, "sample-store"),
                    data=None,
                ),
                dmc.Text(
                    id=aio_id("TrajectoryModal", self.instance_id, "info"),
                    size="sm",
                    c="gray",
                    mb="md",
                ),
                dmc.Group(
                    [
                        dmc.Select(
                            id=aio_id("TrajectoryModal", self.instance_id, "state-select"),
                            label="State Variable",
                            data=state_options,  # type: ignore[arg-type]
                            value="0",
                            w=150,
                        ),
                    ],
                    gap="md",
                    mb="md",
                ),
                dmc.Text("Time Range", size="sm", fw="normal"),
                dmc.RangeSlider(
                    id=aio_id("TrajectoryModal", self.instance_id, "time-range"),
                    min=t_min,
                    max=t_max,
                    value=[t_min, default_t_max],  # type: ignore[arg-type]
                    step=(t_max - t_min) / 100 if t_max > t_min else 0.01,
                    minRange=(t_max - t_min) / 50 if t_max > t_min else 0.01,
                    marks=[
                        {"value": t_min, "label": f"{t_min:.1f}"},
                        {"value": t_max, "label": f"{t_max:.1f}"},
                    ],
                    mb="md",
                ),
                dcc.Graph(
                    id=aio_id("TrajectoryModal", self.instance_id, "plot"),
                    figure=_build_empty_figure(),
                    style={"height": "40vh"},
                    config={"displayModeBar": True},
                ),
            ],
        )

    def build_time_series_figure(
        self,
        sample_idx: int | None = None,
        state_var: int = 0,
        time_span: tuple[float, float] | None = None,
    ) -> go.Figure:
        """
        Build time series plot for a selected sample.

        :param sample_idx: Index of the sample to plot.
        :param state_var: State variable index to display.
        :param time_span: Optional (t_min, t_max) to limit time range.
        :return: Plotly figure with trajectory.
        """
        if self.bse.solution is None:
            return _build_empty_figure()

        fig = go.Figure()

        if sample_idx is not None:
            t = self.bse.solution.time.cpu().numpy()
            y = self.bse.solution.y.cpu().numpy()

            if time_span is not None:
                mask = (t >= time_span[0]) & (t <= time_span[1])
                t = t[mask]
                y = y[:, :, mask]

            trajectory = y[sample_idx, state_var, :]

            valid_mask = np.isfinite(trajectory)
            t = t[valid_mask]
            trajectory = trajectory[valid_mask]
            label = (
                self.bse.solution.labels[sample_idx]
                if self.bse.solution.labels is not None
                else "Unknown"
            )

            fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=t,
                    y=trajectory,
                    mode="lines",
                    name=f"Sample {sample_idx} ({label})",
                    line={"color": get_color(0), "width": 2},
                )
            )

            fig.update_layout(title=f"Trajectory: {self.get_state_label(state_var)}")  # pyright: ignore[reportUnknownMemberType]
        else:
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="Click a point in the scatter plot to view its trajectory",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            fig.update_layout(title="Trajectory Viewer")  # pyright: ignore[reportUnknownMemberType]

        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            xaxis_title="Time",
            yaxis_title=self.get_state_label(state_var),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig


def _build_empty_figure() -> go.Figure:
    """Build empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
        text="Click a point in the scatter plot to view its trajectory",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
        title="Trajectory Viewer",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


from dash import MATCH  # noqa: E402


@callback(
    Output(aio_id("TrajectoryModal", MATCH, "plot"), "figure"),
    [
        Input(aio_id("TrajectoryModal", MATCH, "sample-store"), "data"),
        Input(aio_id("TrajectoryModal", MATCH, "state-select"), "value"),
        Input(aio_id("TrajectoryModal", MATCH, "time-range"), "value"),
    ],
    State(aio_id("TrajectoryModal", MATCH, "plot"), "id"),
)
def update_trajectory_plot_aio(
    sample_data: dict[str, Any] | None,
    state_var: str,
    time_range: list[float],
    plot_id: dict[str, Any],
) -> go.Figure:
    """
    Update trajectory plot when sample or controls change.

    Uses MATCH pattern to target only the specific modal instance.
    The State parameter captures the aio_id for retrieving the correct BSE instance.
    """
    if sample_data is None:
        return _build_empty_figure()

    instance_id = plot_id["aio_id"]
    instance = TrajectoryModalAIO.get_instance(instance_id)
    if instance is None:
        return _build_empty_figure()

    sample_idx = sample_data.get("sample_idx")
    if sample_idx is None:
        return _build_empty_figure()

    return instance.build_time_series_figure(
        sample_idx=sample_idx,
        state_var=int(state_var),
        time_span=(time_range[0], time_range[1]),
    )
