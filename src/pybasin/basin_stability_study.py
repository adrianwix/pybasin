"""Parameter Study Basin Stability Estimator."""

import gc
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler
from pybasin.study_params import StudyParams
from pybasin.template_integrator import TemplateIntegrator
from pybasin.types import StudyResult
from pybasin.utils import NumpyEncoder, generate_filename, resolve_folder

warnings.filterwarnings("ignore", message="os.fork\\(\\) was called")

logger = logging.getLogger(__name__)


class BasinStabilityStudy:
    """
    Basin Stability Study.
    """

    def __init__(
        self,
        n: int,
        ode_system: ODESystemProtocol,
        sampler: Sampler,
        solver: SolverProtocol,
        feature_extractor: FeatureExtractor,
        estimator: Any,
        study_params: StudyParams,
        template_integrator: TemplateIntegrator | None = None,
        compute_orbit_data: list[int] | bool = True,
        output_dir: str | Path | None = "results",
        verbose: bool = False,
    ):
        """
        Initialize the Basin Stability Study.

        Sets up the estimator for a parameter study where one or more parameters are
        systematically varied across multiple values. Parameters can be in any component
        (ODE system, sampler, solver, feature extractor, or predictor).

        :param n: Number of initial conditions (samples) to generate for each parameter value.
        :param ode_system: The ODE system model (ODESystem or JaxODESystem).
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system (Solver or JaxSolver).
        :param feature_extractor: The FeatureExtractor object to extract features from trajectories.
        :param estimator: Any sklearn-compatible estimator (classifier or clusterer).
        :param study_params: Parameter study specification (SweepStudyParams, GridStudyParams, etc.).
        :param template_integrator: Template integrator for supervised classifiers.
        :param compute_orbit_data: Enable orbit data computation for orbit diagram plotting:
                         - ``True`` (default): Compute for all state dimensions.
                         - ``False``: Disabled.
                         - ``list[int]``: Compute for specific state indices (e.g., ``[0, 1]``).
        :param output_dir: Directory path for saving results (JSON, plots), or None to disable.
        :param verbose: If True, show detailed logs from BasinStabilityEstimator instances.
                        If False, suppress INFO logs to reduce output clutter during parameter sweeps.
        """
        self.n = n
        self.ode_system = ode_system
        self.sampler = sampler
        self.solver = solver
        self.feature_extractor = feature_extractor
        self.estimator = estimator
        self.study_params = study_params
        self.template_integrator = template_integrator
        self.compute_orbit_data = compute_orbit_data
        self.output_dir = output_dir
        self.verbose = verbose

        self.results: list[StudyResult] = []

    @property
    def studied_parameter_names(self) -> list[str]:
        """Names of the parameters varied in this study.

        :return: List of parameter names extracted from the study labels.
        """
        if not self.results:
            return []
        first_label = self.results[0]["study_label"]
        return list(first_label.keys())

    def _suppress_verbose_logs(self) -> dict[str, list[logging.Filter]]:
        """Suppress verbose logs from BSE and solver loggers.

        The timing summary is emitted by ``pybasin.step_timer`` and is left
        unfiltered so it always appears. All other INFO logs from
        ``BasinStabilityEstimator`` and related loggers are suppressed.

        :return: Mapping of logger names to the filters that were added.
        """
        added_filters: dict[str, list[logging.Filter]] = {}

        if self.verbose:
            return added_filters

        suppress_all = logging.Filter(name="__block_all__")

        for name in (
            "pybasin.basin_stability_estimator",
            "pybasin.predictors.base",
            "pybasin.solvers.jax_solver",
        ):
            log = logging.getLogger(name)
            log.addFilter(suppress_all)
            added_filters[name] = [suppress_all]

        return added_filters

    def _restore_log_levels(self, added_filters: dict[str, list[logging.Filter]]) -> None:
        """Remove filters added by :meth:`_suppress_verbose_logs`.

        :param added_filters: Mapping returned by ``_suppress_verbose_logs``.
        """
        for logger_name, filters in added_filters.items():
            log = logging.getLogger(logger_name)
            for f in filters:
                log.removeFilter(f)

    def run(
        self,
    ) -> list[StudyResult]:
        """
        Estimate basin stability for each parameter combination in the study.

        Performs basin stability estimation by systematically varying parameters
        across the provided combinations. For each configuration:

        1. Applies all parameter assignments from the RunConfig
        2. Creates a new BasinStabilityEstimator instance
        3. Estimates basin stability and computes error estimates
        4. Stores results including basin stability values, errors, and solution metadata

        Uses GPU acceleration automatically when available for significant performance gains.
        Memory is explicitly freed after each iteration to prevent accumulation.

        :return: List of StudyResult dicts, one per parameter combination, containing
                 study_label, basin_stability, errors, n_samples, labels, and
                 orbit_data (if ``compute_orbit_data`` is set).
        """
        self.results = []

        total_runs = len(self.study_params)

        logger.info("\n" + "=" * 60)
        logger.info("PARAMETER STUDY: %d runs", total_runs)
        logger.info("=" * 60)

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Computing device: %s\n", device)

        for idx, run_config in enumerate(self.study_params, 1):
            context: dict[str, Any] = {
                "n": self.n,
                "ode_system": self.ode_system,
                "sampler": self.sampler,
                "solver": self.solver,
                "feature_extractor": self.feature_extractor,
                "estimator": self.estimator,
                "template_integrator": self.template_integrator,
            }

            # Apply all parameter assignments for this run
            for assignment in run_config.assignments:
                context["_param_value"] = assignment.value
                exec_code = f"{assignment.name} = _param_value"
                eval(compile(exec_code, "<string>", "exec"), context, context)

            logger.info("-" * 60)
            label_str = ", ".join(f"{k}={v}" for k, v in run_config.study_label.items())
            logger.info("[%d/%d] %s", idx, total_runs, label_str)
            logger.info("-" * 60)

            # Use sampler's n_samples if available (for CsvSampler), otherwise use self.n
            n_samples = (
                context["sampler"].n_samples
                if hasattr(context["sampler"], "n_samples")
                else context["n"]
            )

            bse = BasinStabilityEstimator(
                n=n_samples,
                ode_system=context["ode_system"],
                sampler=context["sampler"],
                solver=context["solver"],
                feature_extractor=context["feature_extractor"],
                predictor=context["estimator"],
                template_integrator=context["template_integrator"],
                feature_selector=None,
                compute_orbit_data=self.compute_orbit_data,
            )

            original_levels = self._suppress_verbose_logs()
            result = bse.estimate_bs()
            self._restore_log_levels(original_levels)

            result: StudyResult = {**result, "study_label": run_config.study_label}
            self.results.append(result)

            # Format basin stability output
            bs_str = ", ".join([f"{k}: {v:.4f}" for k, v in result["basin_stability"].items()])
            logger.info("Result: {%s}", bs_str)

            # Explicitly free memory after each iteration
            del bse
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.results

    def save(self) -> None:
        """
        Save the basin stability results to a JSON file.
        Handles numpy arrays and Solution objects by converting them to standard Python types.
        """
        if len(self.results) == 0:
            raise ValueError("No results to save. Please run run() first.")

        if self.output_dir is None:
            raise ValueError("output_dir is not defined.")

        full_folder = resolve_folder(self.output_dir)
        file_name = generate_filename("parameter_study_results", "json")
        full_path = full_folder / file_name

        def format_ode_system(ode_str: str) -> list[str]:
            lines = ode_str.strip().split("\n")
            formatted_lines = [" ".join(line.split()) for line in lines]
            return formatted_lines

        region_of_interest = " X ".join(
            [
                f"[{min_val}, {max_val}]"
                for min_val, max_val in zip(
                    self.sampler.min_limits, self.sampler.max_limits, strict=True
                )
            ]
        )

        parameter_study = [
            {"study_label": r["study_label"], "basin_of_attraction": r["basin_stability"]}
            for r in self.results
        ]

        studied_params = self.studied_parameter_names

        results: dict[str, Any] = {
            "studied_parameters": studied_params,
            "parameter_study": parameter_study,
            "region_of_interest": region_of_interest,
            "sampling_points": self.n,
            "sampling_method": self.sampler.__class__.__name__,
            "solver": self.solver.__class__.__name__,
            "estimator": self.estimator.__class__.__name__,
            "ode_system": format_ode_system(self.ode_system.get_str()),
        }

        with open(full_path, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        logger.info("Results saved to %s", full_path)
