import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from sklearn.base import BaseEstimator, is_classifier, is_regressor  # type: ignore[import-untyped]

from pybasin.constants import DEFAULT_STEADY_FRACTION
from pybasin.feature_extractors import TorchFeatureExtractor
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_selector.default_feature_selector import DefaultFeatureSelector
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer
from pybasin.protocols import (
    UNSET,
    FeatureNameAware,
    FeatureSelectorProtocol,
    ODESystemProtocol,
    SklearnClassifier,
    SklearnClusterer,
    SolverProtocol,
)
from pybasin.sampler import Sampler
from pybasin.solution import Solution
from pybasin.solvers.torchdiffeq_solver import TorchDiffEqSolver
from pybasin.step_timer import StepTimer
from pybasin.study_params import RunConfig, StudyParams
from pybasin.template_integrator import TemplateIntegrator
from pybasin.ts_torch.settings import DEFAULT_TORCH_FC_PARAMETERS
from pybasin.types import ErrorInfo, StudyResult
from pybasin.utils import (
    NumpyEncoder,
    OrbitData,
    extract_orbit_data,
    generate_filename,
    get_feature_names,
    resolve_folder,
)

if TYPE_CHECKING:
    from pybasin.solvers.jax_ode_system import JaxODESystem
    from pybasin.solvers.jax_solver import JaxSolver

try:
    from pybasin.solvers.jax_ode_system import JaxODESystem  # noqa: F811
    from pybasin.solvers.jax_solver import JaxSolver  # noqa: F811

    _jax_available = True
except ImportError:
    _jax_available = False

warnings.filterwarnings("ignore", message="os.fork\\(\\) was called")

logger = logging.getLogger(__name__)


class BasinStabilityEstimator:
    """
    Core class for basin stability analysis.

    Configures the analysis with an ODE system, sampler, and solver,
    and provides methods to estimate basin stability and save results.

    :ivar result: The last computed StudyResult, or None if run() has not been called.
    :ivar y0: Initial conditions tensor.
    :ivar solution: Solution instance containing trajectory and analysis results.
    """

    def __init__(
        self,
        ode_system: ODESystemProtocol,
        sampler: Sampler,
        n: int = 10_000,
        solver: SolverProtocol | None = None,
        feature_extractor: FeatureExtractor | None = None,
        predictor: BaseEstimator | None = None,
        template_integrator: TemplateIntegrator | None = None,
        feature_selector: FeatureSelectorProtocol | None = UNSET,  # type: ignore[assignment]
        detect_unbounded: bool = True,
        compute_orbit_data: list[int] | bool = False,
        output_dir: str | Path | None = None,
    ):
        """
        Initialize the BasinStabilityEstimator.

        :param n: Number of initial conditions (samples) to generate.
        :param ode_system: The ODE system model (ODESystem or JaxODESystem).
        :param sampler: The Sampler object to generate initial conditions.
        :param solver: The Solver object to integrate the ODE system (Solver or JaxSolver).
                      If None, automatically instantiates JaxSolver for JaxODESystem or
                      TorchDiffEqSolver for ODESystem with t_span=(0, 1000), t_steps=1000,
                      t_eval=(850, 1000) (saves only the steady-state window), and device
                      from sampler.
        :param feature_extractor: The FeatureExtractor object to extract features from trajectories.
                                 If None, defaults to TorchFeatureExtractor with minimal+dynamical features.
        :param predictor: Any sklearn-compatible estimator (classifier or clusterer).
                         Classifiers (``is_classifier(predictor)`` is True) require a
                         ``template_integrator`` for supervised learning. Clusterers
                         (``is_clusterer(predictor)`` is True) work unsupervised.
                         Regressors are rejected. If None, defaults to
                         ``HDBSCANClusterer(auto_tune=True, assign_noise=True)``.
        :param template_integrator: Template integrator for supervised classifiers.
                                   Required when ``predictor`` is a classifier. Holds template
                                   initial conditions, labels, and ODE params for training.
        :param feature_selector: Feature filtering sklearn transformer with get_support() method.
                                Defaults to DefaultFeatureSelector(). Pass None to disable filtering.
                                Accepts any sklearn transformer (VarianceThreshold, SelectKBest, etc.) or Pipeline.
        :param detect_unbounded: Enable unboundedness detection before feature extraction (default: True).
                                Only activates when solver has event_fn configured (e.g., JaxSolver with event_fn).
                                When enabled, unbounded trajectories are separated and labeled as "unbounded"
                                before feature extraction to prevent imputed Inf values from contaminating features.
        :param compute_orbit_data: Enable orbit data computation for orbit diagram plotting:
                         - ``False`` (default): Disabled.
                         - ``True``: Compute for all state dimensions.
                         - ``list[int]``: Compute for specific state indices (e.g., ``[0, 1]``).
        :param output_dir: Directory path for saving results (JSON, Excel, plots), or None to disable.
        :raises TypeError: If ``predictor`` is a regressor.
        :raises ValueError: If ``predictor`` is a classifier but no ``template_integrator`` is provided.
        """
        self.n = int(n)
        self.ode_system = ode_system
        self.sampler = sampler
        self.output_dir = output_dir

        if solver is not None:
            self.solver = solver
        elif _jax_available and isinstance(ode_system, JaxODESystem):
            self.solver = JaxSolver(
                t_span=(0, 1000),
                t_steps=1000,
                t_eval=(DEFAULT_STEADY_FRACTION * 1000, 1000),
                device=str(sampler.device),
            )
        else:
            self.solver = TorchDiffEqSolver(
                t_span=(0, 1000),
                t_steps=1000,
                t_eval=(DEFAULT_STEADY_FRACTION * 1000, 1000),
                device=str(sampler.device),
            )

        # Initialize feature selector
        if feature_selector is UNSET:
            # Default: use feature filtering with default thresholds
            self.feature_selector: FeatureSelectorProtocol | None = DefaultFeatureSelector()
        else:
            # User explicitly set it (could be None to disable, or a custom selector)
            self.feature_selector = feature_selector

        self._feature_names: list[str] | None = None

        # Unboundedness detection: enabled only if detect_unbounded=True AND solver is JaxSolver with event_fn
        self.detect_unbounded = (
            detect_unbounded
            and _jax_available
            and isinstance(self.solver, JaxSolver)
            and self.solver.event_fn is not None
        )

        if feature_extractor is None:
            device_str = str(getattr(self.solver, "_device_str", "cpu"))
            feature_extractor = TorchFeatureExtractor(
                features=DEFAULT_TORCH_FC_PARAMETERS,
                device=device_str,  # type: ignore[arg-type]
            )

        self.feature_extractor = feature_extractor

        if predictor is not None and is_regressor(predictor):
            raise TypeError(
                f"Regressors are not supported as predictors. "
                f"Got {type(predictor).__name__}. Use a classifier or clusterer instead."
            )

        if predictor is None:
            predictor = HDBSCANClusterer(auto_tune=True, assign_noise=True)
        self.predictor = predictor

        if is_classifier(self.predictor) and template_integrator is None:
            raise ValueError(
                f"Classifier {type(self.predictor).__name__} requires a template_integrator "
                "with template initial conditions and labels for supervised learning."
            )
        self.template_integrator = template_integrator

        self.compute_orbit_data = compute_orbit_data

        self._bs_vals: dict[str, float] | None = None
        self._result: StudyResult | None = None
        self.y0: torch.Tensor | None = None
        self.solution: Solution | None = None
        self.solutions: list[Solution] | None = None

    @property
    def result(self) -> StudyResult | None:
        """The last computed StudyResult, or None if run() has not been called."""
        return self._result

    def run(self, parallel_integration: bool = True) -> StudyResult:
        """
        Estimate basin stability by:
            1. Generating initial conditions using the sampler.
            2. Integrating the ODE system for each sample (in parallel) to produce a Solution.
            3. Extracting features from each Solution.
            4. Clustering/classifying the feature space.
            5. Computing the fraction of samples in each basin.

        This method sets:
            - self.y0
            - self.solution

        :param parallel_integration: If True and using a supervised classifier with template
                                     integrator, run main and template integration in parallel.
        :return: A StudyResult with basin stability values, errors, labels, and orbit data.
        """
        return self._run_basin_stability(parallel_integration=parallel_integration)[0]

    def run_parameter_study(self, study_params: StudyParams) -> list[StudyResult]:
        """Run a batched parameter study with vectorized integration and feature extraction.

        :param study_params: Parameter study specification.
        :return: List of StudyResult, one per parameter combination, in iteration order.
        """
        return self._run_basin_stability(study_params=study_params)

    def _run_basin_stability(
        self,
        study_params: StudyParams | None = None,
        parallel_integration: bool = True,
    ) -> list[StudyResult]:
        """Unified basin stability pipeline for single-run (P=1) and parameter-study (P>1) modes.

        Integration and feature extraction are vectorised over all B×P trajectories in
        one solver call.  Only the inexpensive per-group filter/classify/BS steps loop
        over P, so the cost scales with B (not BxP) for the expensive operations.

        :param study_params: Parameter study spec, or ``None`` for a single baseline run.
        :param parallel_integration: If ``True``, run template and main integration concurrently.
        :return: List of one StudyResult per parameter combination (length 1 for single run).
        """
        timer = StepTimer()
        timer.start()

        # Build params grid; n_configs=1 passes params=None so the solver uses default ODE params
        run_configs: list[RunConfig] = []
        params_grid: torch.Tensor | None = None
        n_configs: int = 1
        study_labels: list[dict[str, Any]] = [{"baseline": True}]
        if study_params is not None:
            run_configs = list(study_params)
            params_grid = self._build_params_grid(run_configs)
            n_configs = len(run_configs)
            study_labels = [rc.study_label for rc in run_configs]

        if n_configs > 1:
            logger.info("Parameter study: %d configurations  (%d samples each)", n_configs, self.n)

        # Step 1: Sampling
        with timer.step("1. Sampling") as step:
            self.y0 = self.sampler.sample(self.n)
            B = len(self.y0)
            step.details["n_samples"] = B

        # Step 2: Integration — all B×P trajectories in one solver call
        # IC-major ordering: index ic*P+p carries (y0[ic], params[p])
        with timer.step("2. Integration") as step:
            t, y_all = self._integrate_trajectories(params_grid, parallel_integration)
            step.details["trajectory_shape"] = str(y_all.shape)
            step.details["mode"] = (
                "parallel"
                if (parallel_integration and self.template_integrator is not None)
                else "sequential"
            )

        # Step 3: Solution covering all B×n_configs trajectories
        y0_all = self.y0.repeat_interleave(n_configs, dim=0) if n_configs > 1 else self.y0
        with timer.step("3. Solution"):
            self.solution = Solution(initial_condition=y0_all, time=t, y=y_all)

        # Step 3b: Unbounded detection across all B×n_configs at once
        unbounded_all: torch.Tensor | None = None
        if self.detect_unbounded:
            with timer.step("3b. Unbounded Detection") as step:
                unbounded_all = self._detect_unbounded_trajectories(y_all)  # (B*n_configs,)
                n_ub_total = int(unbounded_all.sum().item())
                step.details["n_unbounded"] = n_ub_total
                step.details["pct"] = f"{n_ub_total / (B * n_configs) * 100:.1f}%"

        # Step 3c: Orbit data for all B×n_configs trajectories (per-group slices taken below)
        if self.compute_orbit_data:
            dof = (
                list(range(y_all.shape[2]))
                if self.compute_orbit_data is True
                else self.compute_orbit_data
            )
            with timer.step("3c. Orbit Data") as step:
                self.solution.orbit_data = extract_orbit_data(t, y_all, dof=dof)
                step.details["dof"] = str(dof)

        # Step 4: Feature extraction for all B×n_configs trajectories at once
        # Unbounded (Inf-padded) trajectories produce Inf features; excluded per group below
        with timer.step("4. Feature Extraction") as step:
            features_all = self.feature_extractor.extract_features(
                self.solution
            )  # (B*n_configs, n_features)
            feature_names = self._get_feature_names()
            step.details["shape"] = str(features_all.shape)

        # Per-group pipeline: filter → (fit classifier) → classify → BS
        # O(n_configs) cheap sklearn/numpy operations; integration and extraction are already done
        results: list[StudyResult] = []
        per_param_solutions: list[Solution] = []
        for p, study_label in enumerate(study_labels):
            # IC-major: group p occupies rows p, p+n_configs, p+2*n_configs, … in the B×n_configs axis
            features_p = features_all[p::n_configs]  # (B, n_features)
            unbounded_p = (
                unbounded_all[p::n_configs] if unbounded_all is not None else None
            )  # (B,) | None
            n_unbounded_p = int(unbounded_p.sum().item()) if unbounded_p is not None else 0
            y_p = y_all[:, p::n_configs, :]  # (T, B, S)

            assert self.y0 is not None
            solution_p, self._bs_vals = self._process_study_label(
                features_p,
                unbounded_p,
                n_unbounded_p,
                B,
                feature_names,
                timer,
                p,
                n_configs,
                y0=self.y0,
                t=t,
                y_p=y_p,
            )
            per_param_solutions.append(solution_p)

            result = StudyResult(
                study_label=study_label,
                basin_stability=self._bs_vals,
                errors=self.get_errors(),
                n_samples=B,
                labels=solution_p.labels.copy() if solution_p.labels is not None else None,
                orbit_data=solution_p.orbit_data,
                initial_condition=self.y0.cpu().numpy(),
            )
            self._result = result
            results.append(result)

        if n_configs == 1:
            self.solution = per_param_solutions[0]
        else:
            self.solutions = per_param_solutions
            self.solution = None

        timer.summary()
        return results

    def _process_study_label(
        self,
        features_p: torch.Tensor,
        unbounded_p: torch.Tensor | None,
        n_unbounded_p: int,
        B: int,
        feature_names: list[str],
        timer: StepTimer,
        p: int,
        n_configs: int,
        y0: torch.Tensor,
        t: torch.Tensor,
        y_p: torch.Tensor,
    ) -> tuple[Solution, dict[str, float]]:
        """Process a single parameter group: filter, classify, and compute basin stability.

        :param features_p: Feature tensor of shape (B, n_features) for this group.
        :param unbounded_p: Boolean mask of shape (B,) for unbounded trajectories, or None.
        :param n_unbounded_p: Number of unbounded trajectories in this group.
        :param B: Total number of samples.
        :param feature_names: List of feature names.
        :param timer: Step timer for profiling.
        :param p: Group index.
        :param n_configs: Total number of parameter configurations.
        :param y0: Initial conditions tensor of shape (B, S).
        :param t: Time points tensor of shape (N,).
        :param y_p: Trajectory tensor of shape (N, B, S) for this parameter group.
        :return: Tuple of (solution, bs_vals). All results are also stored on the solution.
        """
        if n_unbounded_p == B:
            unbounded_labels = np.array(["unbounded"] * B, dtype=object)
            solution_p = Solution(initial_condition=y0, time=t, y=y_p)
            solution_p.set_labels(unbounded_labels)
            return solution_p, {"unbounded": 1.0}

        # Exclude unbounded rows before sklearn operations (Inf features corrupt fitting)
        bounded_p: torch.Tensor = torch.ones(B, dtype=torch.bool, device=features_p.device)
        if unbounded_p is not None and n_unbounded_p > 0:
            bounded_p = ~unbounded_p
        features_p_bounded = features_p[bounded_p]

        # Step 5: Feature filtering (per group — feature variance may differ by regime)
        p_suffix = f" [p={p}]" if n_configs > 1 else ""
        with timer.step("5. Feature Filtering" + p_suffix) as step:
            features_p_filtered, filtered_names = self._apply_feature_filtering(
                features_p_bounded, feature_names
            )
            if self.feature_selector is not None:
                self._feature_names = filtered_names
            step.details["n_features"] = features_p_filtered.shape[1]

        # Step 5b: Fit supervised classifier on template (selector already fitted above)
        if self.template_integrator is not None and is_classifier(self.predictor):
            with timer.step("5b. Classifier Fit" + p_suffix):
                X_train, y_train = self.template_integrator.get_training_data(
                    self.feature_extractor,
                    feature_selector=self.feature_selector,
                )
                cast(SklearnClassifier, self.predictor).fit(X_train, y_train)

        if hasattr(self.predictor, "set_feature_names"):
            cast(FeatureNameAware, self.predictor).set_feature_names(filtered_names)

        # Step 6: Classify bounded features; reconstruct full label array
        with timer.step("6. Classification" + p_suffix) as step:
            features_np = features_p_filtered.detach().cpu().numpy()
            bounded_labels: np.ndarray
            if is_classifier(self.predictor):
                bounded_labels = cast(SklearnClassifier, self.predictor).predict(features_np)
            else:
                bounded_labels = cast(SklearnClusterer, self.predictor).fit_predict(features_np)

            labels_p = bounded_labels
            if n_unbounded_p > 0 and unbounded_p is not None:
                labels_p = np.empty(B, dtype=object)
                labels_p[unbounded_p.cpu().numpy()] = "unbounded"
                labels_p[bounded_p.cpu().numpy()] = bounded_labels
            step.details["n_labels"] = len(labels_p)

        # Step 7: Basin stability
        with timer.step("7. BS Computation" + p_suffix) as step:
            bs_vals = self._compute_bs(labels_p)
            for label, val in bs_vals.items():
                step.details[label] = f"{val * 100:.2f}%"

        # Orbit data: slice group p's entries from the full B×n_configs orbit tensor
        # (self.solution still holds the temporary batch solution at this point)
        orbit_data_p: OrbitData | None = None
        if self.solution is not None and self.solution.orbit_data is not None:
            od = self.solution.orbit_data
            orbit_data_p = OrbitData(
                peak_values=od.peak_values[:, p::n_configs, :],
                peak_counts=od.peak_counts[p::n_configs, :],
                dof_indices=od.dof_indices,
                time_steady=od.time_steady,
            )

        # Build the per-parameter Solution with all properties set
        solution_p = Solution(initial_condition=y0, time=t, y=y_p)
        solution_p.set_extracted_features(features_p_bounded, feature_names)
        solution_p.set_features(features_p_filtered, filtered_names)
        solution_p.set_labels(labels_p)
        solution_p.orbit_data = orbit_data_p

        return solution_p, bs_vals

    def _build_params_grid(self, run_configs: list[RunConfig]) -> torch.Tensor:
        """Build a ``(P, n_params)`` parameter tensor from run configurations.

        Column order follows the TypedDict field declaration order of ``ode_system.params``.

        :param run_configs: List of RunConfig objects from a StudyParams iterator.
        :return: Tensor of shape ``(P, n_params)`` where each row is one parameter combination.
        """
        params_dict = cast(dict[str, Any], self.ode_system.params)
        param_names = list(params_dict.keys())
        base_vals: list[float] = [float(v) for v in params_dict.values()]
        rows: list[torch.Tensor] = []
        for rc in run_configs:
            row = torch.tensor(base_vals)
            for key, val in rc.study_label.items():
                row[param_names.index(key)] = float(val)
            rows.append(row)
        return torch.stack(rows)

    def _integrate_trajectories(
        self, params: torch.Tensor | None, parallel: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the ODE system, optionally running template integration concurrently.

        :param params: Parameter grid of shape ``(P, n_params)``, or ``None`` for default params.
        :param parallel: If ``True`` and template_integrator is set, run both concurrently.
        :return: Tuple ``(t, y)`` where y has shape ``(t_steps, B*P, n_dims)``.
        """
        assert self.y0 is not None
        if parallel and self.template_integrator is not None:
            with ThreadPoolExecutor(max_workers=2) as executor:
                main_future = executor.submit(
                    self.solver.integrate,
                    self.ode_system,
                    self.y0,
                    params,  # type: ignore[arg-type]
                )
                template_future = executor.submit(
                    self.template_integrator.integrate,
                    self.solver,
                    self.ode_system,
                )
                t, y = main_future.result()
                template_future.result()
        else:
            if self.template_integrator is not None:
                self.template_integrator.integrate(solver=self.solver, ode_system=self.ode_system)
            t, y = self.solver.integrate(self.ode_system, self.y0, params)  # type: ignore[arg-type]
        return t, y

    def _detect_unbounded_trajectories(self, y: torch.Tensor) -> torch.Tensor:
        """Detect unbounded trajectories based on Inf values.

        When JAX Diffrax integration stops due to an event, remaining timesteps are filled with Inf.

        :param y: Trajectory tensor of shape (N, B, S) where N=timesteps, B=batch, S=states.
        :return: Boolean tensor of shape (B,) indicating unbounded trajectories.
        """
        return torch.isinf(y).any(dim=(0, 2))

    def _get_feature_names(self) -> list[str]:
        """Get feature names from extractor.

        :return: List of feature names.
        """
        return self.feature_extractor.feature_names

    def _apply_feature_filtering(
        self, features: torch.Tensor, feature_names: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        """Apply feature filtering using the configured selector.

        :param features: Feature tensor of shape (n_samples, n_features).
        :param feature_names: List of feature names.
        :return: Tuple of (filtered features tensor, filtered feature names).
        :raises ValueError: If filtering removes all features.
        """
        if self.feature_selector is None:
            return features, feature_names

        # Convert to numpy for sklearn
        features_np = features.detach().cpu().numpy()

        # Apply filtering
        features_filtered_np = cast(
            np.ndarray[Any, np.dtype[np.floating[Any]]],
            self.feature_selector.fit_transform(features_np),  # type: ignore[union-attr]
        )

        # Check if any features remain
        if int(features_filtered_np.shape[1]) == 0:
            raise ValueError(
                f"Feature filtering removed all {features_np.shape[1]} features. "
                "Consider lowering variance_threshold or correlation_threshold."
            )

        # Get filtered feature names using utility function
        filtered_names = get_feature_names(self.feature_selector, feature_names)

        # Convert back to tensor
        features_filtered = torch.from_numpy(features_filtered_np).to(  # type: ignore[arg-type]
            dtype=features.dtype, device=features.device
        )

        # Log filtering stats
        n_original: int = int(features_np.shape[1])
        n_filtered: int = int(features_filtered_np.shape[1])
        reduction_pct: float = float((1 - n_filtered / n_original) * 100)
        logger.info(
            "  Feature Filtering: %d → %d features (%.1f%% reduction)",
            n_original,
            n_filtered,
            reduction_pct,
        )

        return features_filtered, filtered_names

    def _compute_bs(self, labels: np.ndarray) -> dict[str, float]:
        """Compute basin stability fractions from labels.

        :param labels: Label array of length n_samples.
        :return: Dict mapping label string to fraction in [0, 1].
        """
        labels_str = np.array([str(label) for label in labels], dtype=object)
        unique_labels, counts = np.unique(labels_str, return_counts=True)
        actual_n = len(labels)
        bs_vals: dict[str, float] = {}
        for label, fraction in zip(unique_labels, counts / float(actual_n), strict=True):
            bs_vals[str(label)] = float(fraction)
        return bs_vals

    def get_errors(self) -> dict[str, ErrorInfo]:
        """
        Compute absolute and relative errors for basin stability estimates.

        The errors are based on Bernoulli experiment statistics:

        - e_abs = sqrt(S_B(A) * (1 - S_B(A)) / N) — absolute standard error
        - e_rel = 1 / sqrt(N * S_B(A)) — relative error

        :return: Dictionary mapping each label to an ErrorInfo with ``e_abs`` and ``e_rel`` keys.
        :raises ValueError: If ``run()`` has not been called yet.
        """
        if self._bs_vals is None:
            raise ValueError("No results available. Please run run() first.")

        errors: dict[str, ErrorInfo] = {}
        n = self.n

        for label, s_b in self._bs_vals.items():
            e_abs = np.sqrt(s_b * (1 - s_b) / n)

            e_rel = 1 / np.sqrt(n * s_b) if s_b > 0 else float("inf")

            errors[label] = ErrorInfo(e_abs=float(e_abs), e_rel=float(e_rel))

        return errors

    def save(self) -> None:
        """
        Save the basin stability results to a JSON file.

        Converts numpy arrays and Solution objects to standard Python types.

        :raises ValueError: If ``run()`` has not been called yet.
        :raises ValueError: If ``output_dir`` is not defined.
        """
        if self._bs_vals is None:
            raise ValueError("No results to save. Please run run() first.")

        if self.output_dir is None:
            raise ValueError("output_dir is not defined.")

        full_folder = resolve_folder(self.output_dir)
        file_name = generate_filename("basin_stability_results", "json")
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

        # Feature selection information
        feature_selection_info: dict[str, Any] = {
            "enabled": self.feature_selector is not None,
        }

        if self.feature_selector is not None:
            feature_selection_info["selector_type"] = type(self.feature_selector).__name__

            # Add feature count information
            if self.solution and self.solution.extracted_features is not None:
                n_extracted = self.solution.extracted_features.shape[1]
                n_filtered = (
                    self.solution.features.shape[1] if self.solution.features is not None else 0
                )
                feature_selection_info["n_features_extracted"] = n_extracted
                feature_selection_info["n_features_filtered"] = n_filtered
                feature_selection_info["reduction_ratio"] = (
                    (1 - n_filtered / n_extracted) if n_extracted > 0 else 0.0
                )

                if self._feature_names:
                    feature_selection_info["feature_names"] = self._feature_names
        else:
            feature_selection_info["selector_type"] = "disabled"

        results: dict[str, Any] = {
            "basin_of_attractions": self._bs_vals,
            "region_of_interest": region_of_interest,
            "sampling_points": self.n,
            "sampling_method": self.sampler.__class__.__name__,
            "solver": self.solver.__class__.__name__,
            "estimator": self.predictor.__class__.__name__,
            "feature_selection": feature_selection_info,
            "ode_system": format_ode_system(self.ode_system.get_str()),
        }

        with open(full_path, "w") as f:
            json.dump(results, f, cls=NumpyEncoder, indent=2)

        logger.info("Results saved to %s", full_path)

    def save_to_excel(self) -> None:
        """
        Save the basin stability results to an Excel file.

        Includes grid samples, labels, and bifurcation amplitudes.

        :raises ValueError: If ``run()`` has not been called yet.
        :raises ValueError: If ``output_dir`` is not defined.
        :raises ValueError: If no solution data is available.
        """
        if self._bs_vals is None:
            raise ValueError("No results to save. Please run run() first.")

        if self.output_dir is None:
            raise ValueError("output_dir is not defined.")

        if self.y0 is None or self.solution is None:
            raise ValueError("No solution data available. Please run run() first.")

        full_folder = resolve_folder(self.output_dir)
        file_name = generate_filename("basin_stability_results", "xlsx")
        full_path = full_folder / file_name

        # Convert tensors to lists for DataFrame
        y0_list: list[Any] = self.y0.detach().cpu().numpy().tolist()

        data: dict[str, Any] = {
            "Grid Sample": [tuple(ic) for ic in y0_list],
            "Labels": self.solution.labels if self.solution.labels is not None else [],
        }

        if self.solution.orbit_data is not None:
            peak_counts = self.solution.orbit_data.peak_counts.cpu().numpy().tolist()
            data["Peak Counts"] = [tuple(pc) for pc in peak_counts]

        df = pd.DataFrame(data)

        df.to_excel(full_path, index=False)  # type: ignore[call-overload]
