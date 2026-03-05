# pyright: basic
"""Interactive web-based plotter using Dash and Plotly with Mantine components."""

import logging
from typing import Literal

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.development.base_component import Component

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.types import (
    InteractivePlotterOptions,
    merge_options,
)

from .basin_stability_aio import BasinStabilityAIO
from .feature_space_aio import FeatureSpaceAIO
from .ids import IDs
from .param_orbit_diagram_aio import ParamOrbitDiagramAIO
from .param_overview_aio import ParamOverviewAIO
from .state_space_aio import StateSpaceAIO
from .study_parameter_manager_aio import StudyParameterManagerAIO
from .template_phase_plot_aio import TemplatePhasePlotAIO
from .template_time_series_aio import TemplateTimeSeriesAIO
from .trajectory_modal_aio import TrajectoryModalAIO

logger = logging.getLogger(__name__)


NavActiveState = Literal["exact", "partial"] | None


class InteractivePlotter:
    """
    Interactive web-based plotter for basin stability visualization.

    Uses Dash with Mantine components for a modern UI and Plotly for
    interactive visualizations. Each page owns its controls, plot, and callbacks.

    :ivar bse: BasinStabilityEstimator instance with computed results.
    :ivar state_labels: Optional mapping of state indices to custom labels.
    :ivar app: Dash application instance.
    """

    # Main layout IDs
    PAGE_CONTAINER = "page-container"
    CURRENT_VIEW = "current-view"
    URL = "url"

    def __init__(
        self,
        bse: BasinStabilityEstimator | BasinStabilityStudy,
        state_labels: dict[int, str] | None = None,
        options: InteractivePlotterOptions | None = None,
    ):
        """
        Initialize the Plotter.

        :param bse: BasinStabilityEstimator or BasinStabilityStudy instance.
        :param state_labels: Optional dict mapping state indices to labels,
                            e.g., {0: "θ", 1: "ω"} for a pendulum system.
        :param options: Optional configuration for default control values.
        """
        self.is_parameter_study = isinstance(bse, BasinStabilityStudy)

        if isinstance(bse, BasinStabilityStudy):
            self.bs_study: BasinStabilityStudy = bse
        else:
            self.bse: BasinStabilityEstimator = bse

        self.state_labels = state_labels or {}
        self.options = merge_options(options)

        # Override initial view for parameter study mode if not explicitly set
        if self.is_parameter_study and options is None:
            self.options["initial_view"] = IDs.PARAM_OVERVIEW

        self.app: Dash | None = None

        if self.is_parameter_study:
            self._validate_bs_study()
            self._init_study_pages()
        else:
            self._validate_bse()
            self._init_bse_pages()

    def _validate_bse(self) -> None:
        """Validate that BSE has required data for plotting."""
        if self.bse.solution is None:
            raise ValueError("No solution available. Run estimate_bs() first.")
        if self.bse.y0 is None:
            raise ValueError("No initial conditions available. Run estimate_bs() first.")
        if self.bse.solution.labels is None:
            raise ValueError("No labels available. Run estimate_bs() first.")
        if self.bse.bs_vals is None:
            raise ValueError("No basin stability values available. Run estimate_bs() first.")

    def _validate_bs_study(self) -> None:
        """Validate that parameter study has required data for plotting."""
        if len(self.bs_study.results) == 0:
            raise ValueError("No results available. Run run() first.")

    def _get_n_states(self) -> int:
        """Get number of state variables."""
        if self.is_parameter_study:
            return self.bs_study.sampler.state_dim
        if self.bse.y0 is None:
            return 0
        return self.bse.y0.shape[1]

    def _compute_param_bse(self, param_idx: int) -> BasinStabilityEstimator:
        """Compute BSE for a specific parameter index.

        :param param_idx: Index of the parameter value to compute.
        :return: Computed BasinStabilityEstimator instance.
        :raises ValueError: If computation fails.
        """
        try:
            # Get the RunConfig for this parameter index from study_params
            run_configs = list(self.bs_study.study_params)
            if param_idx >= len(run_configs):
                raise ValueError(f"Parameter index {param_idx} out of range")
            run_config = run_configs[param_idx]

            label_str = ", ".join(f"{k}={v}" for k, v in run_config.study_label.items())
            logger.info(f"Computing BSE for {label_str} (index {param_idx})")

            # Build context with all study components
            context: dict[str, object] = {
                "n": self.bs_study.n,
                "ode_system": self.bs_study.ode_system,
                "sampler": self.bs_study.sampler,
                "solver": self.bs_study.solver,
                "feature_extractor": self.bs_study.feature_extractor,
                "estimator": self.bs_study.estimator,
                "template_integrator": self.bs_study.template_integrator,
            }

            # Apply all parameter assignments (uses full paths like ode_system.ode_params['T'])
            for assignment in run_config.assignments:
                context["_param_value"] = assignment.value
                exec_code = f"{assignment.name} = _param_value"
                exec(compile(exec_code, "<string>", "exec"), context, context)

            # Create and run BSE with updated context
            bse = BasinStabilityEstimator(
                n=self.bs_study.n,
                ode_system=context["ode_system"],  # type: ignore[arg-type]
                sampler=context["sampler"],  # type: ignore[arg-type]
                solver=context["solver"],  # type: ignore[arg-type]
                feature_extractor=context["feature_extractor"],  # type: ignore[arg-type]
                predictor=context["estimator"],  # type: ignore[arg-type]
                template_integrator=context["template_integrator"],  # type: ignore[arg-type]
                feature_selector=None,
            )

            bse.estimate_bs()
            logger.info(f"BSE computation complete for {label_str}")

            return bse

        except Exception as e:
            logger.error(f"Failed to compute BSE for parameter index {param_idx}: {e}")
            raise ValueError(f"BSE computation failed: {e}") from e

    def _init_study_pages(self) -> None:
        """Initialize parameter study mode pages with AIO components."""
        self.param_manager = StudyParameterManagerAIO(
            self.bs_study, self.state_labels, self._compute_param_bse
        )
        self.param_overview = ParamOverviewAIO(self.bs_study, "as-overview", self.state_labels)
        self.param_orbit_diagram = ParamOrbitDiagramAIO(
            self.bs_study, "as-orbit-diagram", self.state_labels
        )

    def _init_bse_pages(self) -> None:
        """Initialize BSE mode pages with AIO components."""
        self.state_space = StateSpaceAIO(
            self.bse, "main-state", self.state_labels, self.options.get("state_space")
        )
        self.feature_space = FeatureSpaceAIO(
            self.bse, "main-feature", self.state_labels, self.options.get("feature_space")
        )
        self.basin_stability = BasinStabilityAIO(self.bse, "main-basin", self.state_labels)
        self.templates_phase_space = TemplatePhasePlotAIO(
            self.bse,
            "main-phase",
            False,
            self.state_labels,
            self.options.get("templates_phase_space"),
        )
        self.templates_time_series = TemplateTimeSeriesAIO(
            self.bse, "main-template", self.state_labels, self.options.get("templates_time_series")
        )

    def _create_layout(self) -> dmc.MantineProvider:
        """Create the Dash app layout with navigation and page container."""
        n_states = self._get_n_states()
        initial_view = self.options.get("initial_view", "basin_stability")

        if self.is_parameter_study:
            nav_items = self._build_study_nav_items()
            # Page content is determined by URL callback - start empty
            initial_page_content = html.Div()
            trajectory_modal_content = html.Div(id="as-modals-container")
        else:
            nav_items = self._build_bse_nav_items(n_states, initial_view)
            page_components = {
                IDs.BASIN_STABILITY: self.basin_stability,
                IDs.STATE_SPACE: self.state_space,
                IDs.FEATURE_SPACE: self.feature_space,
                IDs.TEMPLATES_PHASE_SPACE: self.templates_phase_space,
                IDs.TEMPLATES_TIME_SERIES: self.templates_time_series,
            }
            component = page_components.get(initial_view, self.basin_stability)
            initial_page_content = component.render()
            trajectory_modal_content = TrajectoryModalAIO(
                self.bse, "main-modal", self.state_labels
            ).render()

        return dmc.MantineProvider(
            forceColorScheme="dark",
            children=[
                dcc.Location(id=self.URL, refresh=False),
                # Full-screen loading overlay that blocks all interaction
                html.Div(
                    id="page-loading-overlay",
                    style={
                        "display": "none",
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "right": 0,
                        "bottom": 0,
                        "backgroundColor": "rgba(0, 0, 0, 0.5)",
                        "zIndex": 9999,
                        "cursor": "wait",
                    },
                    children=[
                        dmc.Center(
                            style={"height": "100vh"},
                            children=[dmc.Loader(size="xl", color="blue")],
                        )
                    ],
                ),
                dmc.AppShell(
                    [
                        dmc.AppShellHeader(
                            dmc.Group(
                                [
                                    dmc.Title("pybasin: Basin Stability Explorer", order=3),
                                ],
                                px="md",
                                h="100%",
                            ),
                        ),
                        dmc.AppShellNavbar(
                            [dmc.ScrollArea(nav_items, type="scroll")],
                            p="md",
                        ),
                        dmc.AppShellMain(
                            [
                                dmc.Container(
                                    [
                                        html.Div(
                                            id=self.PAGE_CONTAINER,
                                            children=initial_page_content,
                                        ),
                                        trajectory_modal_content,
                                    ],
                                    fluid=True,
                                    p=0,
                                ),
                            ],
                        ),
                    ],
                    header={"height": 60},
                    navbar={"width": 220, "breakpoint": "sm"},
                    padding=0,
                ),
                dcc.Store(id=self.CURRENT_VIEW, data=initial_view),
            ],
        )

    def _build_bse_nav_items(self, n_states: int, initial_view: str) -> list:
        """Build navigation items for standard BSE mode."""
        return [
            dmc.NavLink(
                label="Basin Stability",
                leftSection=html.Span("📊"),
                id="nav-basin-stability",
                active=self._nav_active_state(initial_view == IDs.BASIN_STABILITY),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="State Space",
                leftSection=html.Span("🎯"),
                id="nav-state-space",
                active=self._nav_active_state(initial_view == IDs.STATE_SPACE),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="Feature Space",
                leftSection=html.Span("📈"),
                id="nav-feature-space",
                active=self._nav_active_state(initial_view == IDs.FEATURE_SPACE),
                n_clicks=0,
            ),
            dmc.Divider(label="Template Trajectories", my="sm"),
            dmc.NavLink(
                label="Phase Space",
                leftSection=html.Span("〰️"),
                id="nav-templates-phase-space",
                active=self._nav_active_state(initial_view == IDs.TEMPLATES_PHASE_SPACE),
                n_clicks=0,
            ),
            dmc.NavLink(
                label="Time Series",
                leftSection=html.Span("📉"),
                id="nav-templates-time-series",
                active=self._nav_active_state(initial_view == IDs.TEMPLATES_TIME_SERIES),
                n_clicks=0,
            ),
        ]

    def _build_study_nav_items(self) -> list:
        """Build navigation items for parameter study mode."""
        study_labels = [r["study_label"] for r in self.bs_study.results]
        param_names = self.bs_study.studied_parameter_names
        param_options = [
            {
                "value": str(i),
                "label": ", ".join(f"{p}={study_labels[i][p]:.4g}" for p in param_names),
            }
            for i in range(len(study_labels))
        ]

        return [
            dmc.Divider(label="Parameter Study", my="sm"),
            dmc.NavLink(
                label="Stability analysis",
                leftSection=html.Span("📊"),
                href="/overview",
                active="exact",
            ),
            dmc.NavLink(
                label="Orbit Diagram",
                leftSection=html.Span("🌀"),
                href="/orbit-diagram",
                active="exact",
            ),
            dmc.Divider(label="Parameter Value", my="sm"),
            dmc.Select(
                id="param-value-selector",
                label="Select Parameter",
                # Type is wrong. Check: https://www.dash-mantine-components.com/components/select
                data=param_options,  # type: ignore[arg-type]
                value="0",
                searchable=True,
                allowDeselect=False,
                mb="md",
            ),
            html.Div(id="param-bse-nav-items"),
        ]

    def run(self, port: int = 8050, debug: bool = False) -> None:
        """
        Launch the interactive plotter as a standalone Dash server.

        :param port: Port to run the server on (default: 8050).
        :param debug: Enable Dash debug mode (default: False).
        """
        self.app = Dash(
            __name__,
            external_stylesheets=[],
            suppress_callback_exceptions=True,
        )

        self.app.layout = self._create_layout()

        # Register navigation callbacks (using @self.app.callback for view switching)
        self._register_navigation_callbacks()

        logger.info("\n🚀 Starting Basin Stability Visualization Server")
        self.app.run(host="0.0.0.0", port=port, debug=debug)

    def _register_navigation_callbacks(self) -> None:
        """Register callbacks for navigation and page switching."""
        if self.app is None:
            return

        if self.is_parameter_study:
            self._register_study_nav_callbacks()
        else:
            self._register_bse_nav_callbacks()

    def _register_bse_nav_callbacks(self) -> None:
        """Register navigation callbacks for standard BSE mode."""
        if self.app is None:
            return

        path_to_view = {
            "/": IDs.BASIN_STABILITY,
            "/basin-stability": IDs.BASIN_STABILITY,
            "/state-space": IDs.STATE_SPACE,
            "/feature-space": IDs.FEATURE_SPACE,
            "/phase": IDs.TEMPLATES_PHASE_SPACE,
            "/time-series": IDs.TEMPLATES_TIME_SERIES,
        }
        view_to_path = {
            IDs.BASIN_STABILITY: "/basin-stability",
            IDs.STATE_SPACE: "/state-space",
            IDs.FEATURE_SPACE: "/feature-space",
            IDs.TEMPLATES_PHASE_SPACE: "/phase",
            IDs.TEMPLATES_TIME_SERIES: "/time-series",
        }

        @self.app.callback(
            Output(self.URL, "pathname"),
            [
                Input("nav-basin-stability", "n_clicks"),
                Input("nav-state-space", "n_clicks"),
                Input("nav-feature-space", "n_clicks"),
                Input("nav-templates-phase-space", "n_clicks"),
                Input("nav-templates-time-series", "n_clicks"),
            ],
            prevent_initial_call=True,
        )
        def update_url(*_args: int) -> str:
            triggered = ctx.triggered_id
            views = {
                "nav-basin-stability": IDs.BASIN_STABILITY,
                "nav-state-space": IDs.STATE_SPACE,
                "nav-feature-space": IDs.FEATURE_SPACE,
                "nav-templates-phase-space": IDs.TEMPLATES_PHASE_SPACE,
                "nav-templates-time-series": IDs.TEMPLATES_TIME_SERIES,
            }
            view = (
                views.get(triggered, IDs.BASIN_STABILITY)
                if isinstance(triggered, str)
                else IDs.BASIN_STABILITY
            )
            return view_to_path.get(view, "/basin-stability")

        @self.app.callback(
            [
                Output(self.CURRENT_VIEW, "data"),
                Output(self.PAGE_CONTAINER, "children"),
                Output("nav-basin-stability", "active"),
                Output("nav-state-space", "active"),
                Output("nav-feature-space", "active"),
                Output("nav-templates-phase-space", "active"),
                Output("nav-templates-time-series", "active"),
            ],
            Input(self.URL, "pathname"),
        )
        def update_page_from_url(
            pathname: str | None,
        ) -> tuple[
            str,
            html.Div,
            NavActiveState,
            NavActiveState,
            NavActiveState,
            NavActiveState,
            NavActiveState,
        ]:
            if pathname is None:
                pathname = "/"
            view = path_to_view.get(pathname, IDs.BASIN_STABILITY)

            page_components = {
                IDs.BASIN_STABILITY: self.basin_stability,
                IDs.STATE_SPACE: self.state_space,
                IDs.FEATURE_SPACE: self.feature_space,
                IDs.TEMPLATES_PHASE_SPACE: self.templates_phase_space,
                IDs.TEMPLATES_TIME_SERIES: self.templates_time_series,
            }
            component = page_components.get(view, self.basin_stability)
            page_layout = component.render()

            view_to_nav = {
                IDs.BASIN_STABILITY: "nav-basin-stability",
                IDs.STATE_SPACE: "nav-state-space",
                IDs.FEATURE_SPACE: "nav-feature-space",
                IDs.TEMPLATES_PHASE_SPACE: "nav-templates-phase-space",
                IDs.TEMPLATES_TIME_SERIES: "nav-templates-time-series",
            }
            active_nav = view_to_nav.get(view, "nav-basin-stability")

            return (
                view,
                page_layout,
                self._nav_active_state(active_nav == "nav-basin-stability"),
                self._nav_active_state(active_nav == "nav-state-space"),
                self._nav_active_state(active_nav == "nav-feature-space"),
                self._nav_active_state(active_nav == "nav-templates-phase-space"),
                self._nav_active_state(active_nav == "nav-templates-time-series"),
            )

    def _register_study_nav_callbacks(self) -> None:
        """Register navigation callbacks for parameter study mode.

        Uses URL-based routing with pattern: /{page}/{param_idx}
        Examples: /overview, /state-space/1, /orbit-diagram
        Study-level pages (overview, orbit-diagram) don't need param_idx.
        BSE pages require param_idx: /state-space/2, /basin-stability/0
        """
        if self.app is None:
            return

        # Valid page names for URL routing
        study_pages = {"overview", "orbit-diagram"}
        bse_pages = {
            "state-space",
            "feature-space",
            "basin-stability",
            "templates-phase-space",
            "templates-time-series",
        }

        def parse_url(pathname: str | None) -> tuple[int, str]:
            """Parse URL into (param_idx, page). Returns (0, 'overview') as default."""
            if not pathname or pathname == "/":
                return 0, "overview"
            parts = pathname.strip("/").split("/")
            if len(parts) >= 1:
                page = parts[0]
                if page in study_pages:
                    return 0, page
                if page in bse_pages:
                    param_idx = int(parts[1]) if len(parts) >= 2 else 0
                    return param_idx, page
            return 0, "overview"

        def build_url(page: str, param_idx: int | None = None) -> str:
            """Build URL from page and optional param_idx."""
            if page in study_pages or param_idx is None:
                return f"/{page}"
            return f"/{page}/{param_idx}"

        # Callback: Selector change -> URL update (triggers loading for BSE pages)
        @self.app.callback(
            Output(self.URL, "pathname"),
            Input("param-value-selector", "value"),
            State(self.URL, "pathname"),
            running=[
                (
                    Output("page-loading-overlay", "style"),
                    {
                        "display": "block",
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "right": 0,
                        "bottom": 0,
                        "backgroundColor": "rgba(0, 0, 0, 0.5)",
                        "zIndex": 9999,
                        "cursor": "wait",
                    },
                    {"display": "none"},
                ),
            ],
            prevent_initial_call=True,
        )
        def update_url_from_selector(
            param_idx_str: str | None,
            current_path: str | None,
        ) -> str:
            current_param_idx, current_page = parse_url(current_path)
            param_idx = int(param_idx_str) if param_idx_str else current_param_idx

            # Only update if param actually changed
            if param_idx == current_param_idx:
                return no_update  # type: ignore[return-value]

            # Study pages don't need param in URL
            if current_page in study_pages:
                return no_update  # type: ignore[return-value]

            return build_url(current_page, param_idx)

        # Callback: URL change -> Update page content and selector
        @self.app.callback(
            [
                Output(self.CURRENT_VIEW, "data"),
                Output(self.PAGE_CONTAINER, "children"),
                Output("param-bse-nav-items", "children"),
                Output("param-value-selector", "value"),
            ],
            Input(self.URL, "pathname"),
            running=[
                (
                    Output("page-loading-overlay", "style"),
                    {
                        "display": "block",
                        "position": "fixed",
                        "top": 0,
                        "left": 0,
                        "right": 0,
                        "bottom": 0,
                        "backgroundColor": "rgba(0, 0, 0, 0.5)",
                        "zIndex": 9999,
                        "cursor": "wait",
                    },
                    {"display": "none"},
                ),
            ],
        )
        def update_page_from_url(
            pathname: str | None,
        ) -> tuple[str, Component, list[Component], str]:
            param_idx, page = parse_url(pathname)

            # Build BSE nav items with href for automatic active state
            bse_nav_items: list[Component] = [
                dmc.Divider(label="Basin Stability Plots", my="sm"),
                dmc.NavLink(
                    label="Basin Stability",
                    leftSection=html.Span("📊"),
                    href=f"/basin-stability/{param_idx}",
                    active="exact",
                ),
                dmc.NavLink(
                    label="State Space",
                    leftSection=html.Span("🎯"),
                    href=f"/state-space/{param_idx}",
                    active="exact",
                ),
                dmc.NavLink(
                    label="Feature Space",
                    leftSection=html.Span("📈"),
                    href=f"/feature-space/{param_idx}",
                    active="exact",
                ),
                dmc.Divider(label="Template Trajectories", my="sm"),
                dmc.NavLink(
                    label="Phase Space",
                    leftSection=html.Span("〰️"),
                    href=f"/templates-phase-space/{param_idx}",
                    active="exact",
                ),
                dmc.NavLink(
                    label="Time Series",
                    leftSection=html.Span("📉"),
                    href=f"/templates-time-series/{param_idx}",
                    active="exact",
                ),
            ]

            # Render the appropriate page
            if page == "overview":
                content = self.param_overview.render()
            elif page == "orbit-diagram":
                content = self.param_orbit_diagram.render()
            elif page in bse_pages:
                pages, _ = self.param_manager.get_or_create_pages(param_idx)
                page_component = pages.get(page)
                content = (
                    page_component.render() if page_component else self.param_overview.render()
                )
            else:
                content = self.param_overview.render()

            return (
                f"{page}/{param_idx}",
                content,
                bse_nav_items,
                str(param_idx),  # Sync selector from URL
            )

    def _nav_active_state(self, is_active: bool) -> NavActiveState:
        return "exact" if is_active else None
