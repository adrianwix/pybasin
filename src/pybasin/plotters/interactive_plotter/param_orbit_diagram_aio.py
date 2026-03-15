"""AIO Parameter Orbit Diagram page showing peak amplitude evolution."""

import contextlib
from collections import defaultdict
from typing import Any

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
from plotly.subplots import (  # pyright: ignore[reportMissingTypeStubs]
    make_subplots,  # pyright: ignore[reportUnknownVariableType]
)

from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.utils import get_color
from pybasin.utils import OrbitData


class ParamOrbitDiagramAIO:
    """
    AIO component for parameter orbit diagram page.

    Shows peak amplitude evolution across parameter sweep, grouped by attractor.
    Supports multi-parameter studies by grouping results along one chosen parameter
    (x-axis) while producing separate curves for each combination of the remaining
    parameters.
    """

    _instances: dict[str, "ParamOrbitDiagramAIO"] = {}

    @classmethod
    def get_instance(cls, instance_id: str) -> "ParamOrbitDiagramAIO | None":
        """Get instance by ID."""
        return cls._instances.get(instance_id)

    def __init__(
        self,
        bs_study: BasinStabilityStudy,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize parameter orbit diagram AIO component.

        :param bs_study: Basin stability study instance.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to labels.
        """
        self.bs_study = bs_study
        self.aio_id = aio_id
        self.state_labels = state_labels or {}
        self._attractor_labels_cache: list[str] | None = None
        self._peak_values_cache: dict[int, np.ndarray[Any, Any]] = {}
        ParamOrbitDiagramAIO._instances[aio_id] = self

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def _get_orbit_data_dofs(self) -> list[int]:
        """Get available DOF indices from orbit data."""
        if not self.bs_study.results:
            return []
        first_orbit_data: OrbitData | None = self.bs_study.results[0].get("orbit_data")
        if first_orbit_data is None:
            return []
        return first_orbit_data.dof_indices

    def _has_orbit_data(self) -> bool:
        """Check if orbit data is available."""
        if not self.bs_study.results:
            return False
        return self.bs_study.results[0].get("orbit_data") is not None

    def get_state_options(self) -> list[dict[str, str]]:
        """Get dropdown options for state variable selection."""
        dof_indices = self._get_orbit_data_dofs()
        return [{"value": str(i), "label": self.get_state_label(i)} for i in dof_indices]

    def get_parameter_names(self) -> list[str]:
        """Get all studied parameter names."""
        return self.bs_study.studied_parameter_names

    def get_parameter_options(self) -> list[dict[str, str]]:
        """Get dropdown options for parameter selection."""
        return [{"value": name, "label": name} for name in self.get_parameter_names()]

    def _get_attractor_labels(self) -> list[str]:
        """Collect all unique attractor labels across every run, sorted."""
        if self._attractor_labels_cache is not None:
            return self._attractor_labels_cache
        labels_set: set[str] = set()
        for r in self.bs_study.results:
            labels_set.update(r["basin_stability"].keys())
        self._attractor_labels_cache = sorted(labels_set)
        return self._attractor_labels_cache

    def _group_by_parameter(self, param_name: str) -> dict[tuple[tuple[str, Any], ...], list[int]]:
        """Group study result indices by the values of all parameters except param_name.

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
        if not self._has_orbit_data():
            return html.Div(
                [
                    dmc.Alert(
                        "No orbit data available. Set compute_orbit_data=True when creating "
                        "BasinStabilityStudy to enable orbit diagram plotting.",
                        title="Orbit Data Required",
                        color="yellow",
                    )
                ],
                style={"padding": "20px"},
            )

        state_options = self.get_state_options()
        param_options = self.get_parameter_options()
        default_param = param_options[0]["value"] if param_options else ""
        default_dofs = state_options[0]["value"] if state_options else "0"

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.Group(
                            [
                                dmc.Select(
                                    id=aio_id("ParamOrbitDiagram", self.aio_id, "x_param"),
                                    label="X-Axis Parameter",
                                    data=param_options,  # type: ignore[arg-type]
                                    value=default_param,
                                    w=200,
                                ),
                                dmc.Select(
                                    id=aio_id("ParamOrbitDiagram", self.aio_id, "dofs"),
                                    label="State Dimension",
                                    data=state_options,  # type: ignore[arg-type]
                                    value=default_dofs,
                                    w=200,
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
                        dcc.Loading(
                            dcc.Graph(
                                id=aio_id("ParamOrbitDiagram", self.aio_id, "plot"),
                                figure=self.build_figure(
                                    x_param=default_param,
                                    selected_dof=int(default_dofs),
                                ),
                                style={"height": "calc(100vh - 190px)"},
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                },
                            ),
                            type="circle",
                            overlay_style={"visibility": "visible", "opacity": 0.3},
                        ),
                    ],
                    p="md",
                    withBorder=True,
                ),
            ]
        )

    def _get_peak_values_cpu(self, result_idx: int) -> np.ndarray[Any, Any] | None:
        """Return peak_values as a CPU numpy array, cached to avoid repeated GPU transfers.

        :param result_idx: Index into bs_study.results.
        :return: Numpy array of shape (max_peaks, n_trajectories, n_dofs), or None.
        """
        if result_idx in self._peak_values_cache:
            return self._peak_values_cache[result_idx]
        orbit_data: OrbitData | None = self.bs_study.results[result_idx].get("orbit_data")
        if orbit_data is None:
            return None
        arr: np.ndarray[Any, Any] = orbit_data.peak_values.cpu().numpy()
        self._peak_values_cache[result_idx] = arr
        return arr

    def build_figure(
        self,
        x_param: str | None = None,
        selected_dof: int | None = None,
    ) -> go.Figure:
        """Build orbit diagram figure showing peak amplitudes per attractor.

        :param x_param: Parameter to use as x-axis. Defaults to first parameter.
        :param selected_dof: State index to plot. Defaults to first available DOF.
        :return: Plotly figure.
        """
        if not self._has_orbit_data():
            return go.Figure()

        params = self.get_parameter_names()
        if not params:
            return go.Figure()

        if x_param is None or x_param not in params:
            x_param = params[0]

        available_dofs = self._get_orbit_data_dofs()
        if selected_dof is None:
            selected_dof = available_dofs[0] if available_dofs else 0

        attractor_labels = self._get_attractor_labels()
        groups = self._group_by_parameter(x_param)

        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[self.get_state_label(selected_dof)],
        )

        # Precompute dof_pos from the first result (consistent across all results).
        first_orbit_data: OrbitData | None = self.bs_study.results[0].get("orbit_data")
        dof_pos: int | None = None
        if first_orbit_data is not None:
            with contextlib.suppress(ValueError):
                dof_pos = first_orbit_data.dof_indices.index(selected_dof)

        if dof_pos is None:
            return fig

        # Accumulate numpy arrays per attractor; concatenate once at the end.
        xs_parts: dict[int, list[np.ndarray[Any, Any]]] = defaultdict(list)
        ys_parts: dict[int, list[np.ndarray[Any, Any]]] = defaultdict(list)

        for _group_key, indices in groups.items():
            for result_idx in indices:
                result = self.bs_study.results[result_idx]
                labels = result.get("labels")
                if labels is None:
                    continue

                x_val: Any = result["study_label"][x_param]
                peak_values_cpu = self._get_peak_values_cpu(result_idx)
                if peak_values_cpu is None:
                    continue

                for a_idx, attractor in enumerate(attractor_labels):
                    attractor_mask: np.ndarray[Any, Any] = labels == attractor
                    if not np.any(attractor_mask):
                        continue

                    peaks_flat = peak_values_cpu[:, attractor_mask, dof_pos].ravel()
                    valid_peaks = peaks_flat[np.isfinite(peaks_flat)]

                    if valid_peaks.size == 0:
                        continue

                    xs_parts[a_idx].append(np.full(valid_peaks.size, x_val))
                    ys_parts[a_idx].append(valid_peaks)

        # One trace per attractor — O(A) traces total.
        for a_idx, attractor in enumerate(attractor_labels):
            if not xs_parts[a_idx]:
                continue
            color = get_color(a_idx)
            fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                go.Scattergl(
                    x=np.concatenate(xs_parts[a_idx]),
                    y=np.concatenate(ys_parts[a_idx]),
                    mode="markers",
                    name=attractor,
                    marker={"color": color, "size": 4, "opacity": 0.6},
                    showlegend=True,
                    legendgroup=attractor,
                ),
                row=1,
                col=1,
            )

        fig.update_xaxes(title=x_param, autorange=True, row=1, col=1)  # pyright: ignore[reportUnknownMemberType]
        fig.update_yaxes(  # pyright: ignore[reportUnknownMemberType]
            title=f"Amplitude ({self.get_state_label(selected_dof)})", autorange=True, row=1, col=1
        )

        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title=f"Orbit Diagram ({x_param})",
            template="plotly_dark",
            autosize=True,
            legend={"title": "Attractor"},
        )

        return fig


@callback(
    Output(aio_id("ParamOrbitDiagram", MATCH, "plot"), "figure"),
    [
        Input(aio_id("ParamOrbitDiagram", MATCH, "x_param"), "value"),
        Input(aio_id("ParamOrbitDiagram", MATCH, "dofs"), "value"),
    ],
    State(aio_id("ParamOrbitDiagram", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_param_orbit_diagram_figure_aio(
    x_param: str,
    selected_dof_str: str,
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update orbit diagram when controls change."""
    instance_id = plot_id["aio_id"]
    instance = ParamOrbitDiagramAIO.get_instance(instance_id)
    if instance is None:
        return go.Figure()

    return instance.build_figure(x_param=x_param, selected_dof=int(selected_dof_str))
