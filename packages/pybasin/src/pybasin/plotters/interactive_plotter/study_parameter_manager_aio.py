"""AIO Parameter Manager with LRU cache for parameter study mode page management."""

from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import cast

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.interactive_plotter.basin_stability_aio import BasinStabilityAIO
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.feature_space_aio import FeatureSpaceAIO
from pybasin.plotters.interactive_plotter.state_space_aio import StateSpaceAIO
from pybasin.plotters.interactive_plotter.template_phase_plot_aio import TemplatePhasePlotAIO
from pybasin.plotters.interactive_plotter.template_time_series_aio import TemplateTimeSeriesAIO


class StudyParameterManagerAIO:
    """
    AIO component managing parameter-specific page sets with LRU cache.

    Creates page instances on-demand when user navigates to a parameter value.
    Maintains LRU cache of max 5 parameter page sets. Shows loading visualization
    during page instantiation (~100ms).
    """

    MAX_CACHE_SIZE = 5

    def __init__(
        self,
        bs_study: BasinStabilityStudy,
        state_labels: dict[int, str] | None = None,
        compute_bse_callback: Callable[[int], BasinStabilityEstimator] | None = None,
    ):
        """
        Initialize parameter study page manager.

        :param bs_study: Basin stability study instance.
        :param state_labels: Optional mapping of state indices to labels.
        :param compute_bse_callback: Optional callback to compute BSE for a parameter index.
        """
        self.bs_study = bs_study
        self.state_labels = state_labels or {}
        self.compute_bse_callback = compute_bse_callback
        self._page_cache: OrderedDict[int, dict[str, BseBasePageAIO]] = OrderedDict()

    def get_or_create_pages(self, param_index: int) -> tuple[Mapping[str, BseBasePageAIO], bool]:
        """
        Get page set for parameter index, creating if needed.

        Uses LRU cache with max 5 entries. When exceeding limit, clears entire
        cache (simplified eviction strategy).

        :param param_index: Index of parameter value.
        :return: Tuple of (page_dict, is_newly_created).
        """
        if param_index in self._page_cache:
            self._page_cache.move_to_end(param_index)
            return self._page_cache[param_index], False

        if len(self._page_cache) >= self.MAX_CACHE_SIZE:
            self._page_cache.clear()

        pages = self._create_pages_for_param(param_index)
        self._page_cache[param_index] = cast(dict[str, BseBasePageAIO], pages)
        return pages, True

    def _create_pages_for_param(self, param_index: int) -> Mapping[str, BseBasePageAIO]:
        """
        Create complete page set for specific parameter index.

        :param param_index: Index of parameter value.
        :return: Dictionary mapping page IDs to AIO page instances.
        """
        if self.compute_bse_callback is not None:
            bse = self.compute_bse_callback(param_index)
        else:
            # Get the RunConfig for this parameter index from study_params
            run_configs = list(self.bs_study.study_params)
            if param_index >= len(run_configs):
                raise ValueError(f"Parameter index {param_index} out of range")
            run_config = run_configs[param_index]

            context: dict[str, object] = {
                "n": self.bs_study.n,
                "ode_system": self.bs_study.ode_system,
                "sampler": self.bs_study.sampler,
                "solver": self.bs_study.solver,
                "feature_extractor": self.bs_study.feature_extractor,
                "estimator": self.bs_study.estimator,
                "template_integrator": self.bs_study.template_integrator,
            }

            # Apply all parameter assignments (uses full paths)
            for assignment in run_config.assignments:
                context["_param_value"] = assignment.value
                exec_code = f"{assignment.name} = _param_value"
                exec(compile(exec_code, "<string>", "exec"), context, context)

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
            bse.run()

        aio_id_base = f"param-{param_index}"

        pages: dict[str, BseBasePageAIO] = {
            "state-space": StateSpaceAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-state-space",
                state_labels=self.state_labels,
            ),
            "feature-space": FeatureSpaceAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-feature-space",
                state_labels=self.state_labels,
            ),
            "basin-stability": BasinStabilityAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-basin-stability",
                state_labels=self.state_labels,
            ),
            "templates-phase-space": TemplatePhasePlotAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-phase",
                is_3d=False,
                state_labels=self.state_labels,
            ),
            "templates-time-series": TemplateTimeSeriesAIO(
                bse=bse,
                aio_id=f"{aio_id_base}-template-ts",
                state_labels=self.state_labels,
            ),
        }

        return cast(Mapping[str, BseBasePageAIO], pages)
