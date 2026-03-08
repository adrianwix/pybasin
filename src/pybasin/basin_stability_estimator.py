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

    :ivar result: The last computed StudyResult, or None if estimate_bs() has not been called.
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

    @property
    def result(self) -> StudyResult | None:
        """The last computed StudyResult, or None if estimate_bs() has not been called."""
        return self._result

    def estimate_bs(self, parallel_integration: bool = True) -> StudyResult:
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
            - self._bs_vals

        :param parallel_integration: If True and using a supervised classifier with template
                                     integrator, run main and template integration in parallel.
        :return: A StudyResult with basin stability values, errors, labels, and orbit data.
        """
        timer = StepTimer()
        timer.start()

        # Step 1: Sampling
        with timer.step("1. Sampling") as step:
            self.y0 = self.sampler.sample(self.n)
            step.details["n_samples"] = len(self.y0)

        # Step 2: Integration
        with timer.step("2. Integration") as step:
            t, y = self._integrate(parallel_integration)
            step.details["trajectory_shape"] = str(y.shape)
            step.details["mode"] = (
                "parallel"
                if (parallel_integration and self.template_integrator is not None)
                else "sequential"
            )

        # Step 3: Solution
        with timer.step("3. Solution") as step:
            self.solution = Solution(initial_condition=self.y0, time=t, y=y)

        # Step 3b: Detect and separate unbounded trajectories
        unbounded_mask: torch.Tensor | None = None
        n_unbounded = 0
        total_samples = len(self.y0)
        original_solution: Solution | None = None

        if self.detect_unbounded:
            with timer.step("3b. Unbounded Detection") as step:
                unbounded_mask = self._detect_unbounded_trajectories(y)
                n_unbounded = int(unbounded_mask.sum().item())
                step.details["n_unbounded"] = n_unbounded
                step.details["pct"] = f"{(n_unbounded / total_samples) * 100:.1f}%"

            if n_unbounded == total_samples:
                self._bs_vals = {"unbounded": 1.0}
                labels: np.ndarray = np.array(["unbounded"] * total_samples, dtype=object)
                self.solution.set_labels(labels)
                timer.summary()
                self._result = StudyResult(
                    study_label={"baseline": True},
                    basin_stability=self._bs_vals,
                    errors=self.get_errors(),
                    n_samples=len(self.y0),
                    labels=labels.copy(),
                    orbit_data=None,
                )
                return self._result

            if n_unbounded > 0:
                bounded_mask = ~unbounded_mask
                original_solution = self.solution
                self.solution = Solution(
                    initial_condition=self.y0[bounded_mask],
                    time=t,
                    y=y[:, bounded_mask, :],
                )

        # Step 3c: Orbit data
        if self.compute_orbit_data:
            dof = (
                list(range(self.solution.y.shape[2]))
                if self.compute_orbit_data is True
                else self.compute_orbit_data
            )
            with timer.step("3c. Orbit Data") as step:
                self.solution.orbit_data = extract_orbit_data(
                    self.solution.time, self.solution.y, dof=dof
                )
                step.details["dof"] = str(dof)

        # Step 4: Feature extraction
        with timer.step("4. Feature Extraction") as step:
            features = self.feature_extractor.extract_features(self.solution)
            feature_names = self._get_feature_names()
            self.solution.set_extracted_features(features, feature_names)
            step.details["shape"] = str(features.shape)

        # Step 5: Feature filtering
        with timer.step("5. Feature Filtering") as step:
            features, feature_names = self._filter_features(features, feature_names)
            step.details["n_features"] = features.shape[1]
            if self._feature_names:
                n_to_show = min(10, len(self._feature_names))
                for i, name in enumerate(self._feature_names[:n_to_show]):
                    logger.info("  %d. %s", i + 1, name)

        # Step 5b: Fit classifier
        if self.template_integrator is not None and is_classifier(self.predictor):
            with timer.step("5b. Classifier Fit") as step:
                X_train, y_train = self.template_integrator.get_training_data(
                    self.feature_extractor,
                    feature_selector=self.feature_selector,
                )
                cast(SklearnClassifier, self.predictor).fit(X_train, y_train)

        final_feature_names = self._feature_names or feature_names
        if hasattr(self.predictor, "set_feature_names"):
            cast(FeatureNameAware, self.predictor).set_feature_names(final_feature_names)

        # Step 6: Classification
        with timer.step("6. Classification") as step:
            labels = self._classify(
                features, unbounded_mask, n_unbounded, total_samples, original_solution
            )
            self.solution.set_labels(labels)
            step.details["n_labels"] = len(labels)

        # Step 7: Basin stability
        with timer.step("7. BS Computation") as step:
            self._bs_vals = self._compute_bs(labels)
            for label, val in self._bs_vals.items():
                step.details[label] = f"{val * 100:.2f}%"

        timer.summary()

        result = StudyResult(
            study_label={"baseline": True},
            basin_stability=self._bs_vals,
            errors=self.get_errors(),
            n_samples=len(self.y0),
            labels=self.solution.labels.copy() if self.solution.labels is not None else None,
            orbit_data=self.solution.orbit_data,
        )
        self._result = result
        return result

    def _integrate(self, parallel: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate the ODE system, optionally in parallel with template integration.

        :param parallel: If True and template_integrator is set, run both integrations concurrently.
        :return: Tuple of (time tensor, trajectory tensor).
        """
        assert self.y0 is not None
        if parallel and self.template_integrator is not None:
            with ThreadPoolExecutor(max_workers=2) as executor:
                main_future = executor.submit(self.solver.integrate, self.ode_system, self.y0)  # type: ignore[arg-type]
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
            t, y = self.solver.integrate(self.ode_system, self.y0)  # type: ignore[arg-type]
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

    def _filter_features(
        self, features: torch.Tensor, feature_names: list[str]
    ) -> tuple[torch.Tensor, list[str]]:
        """Apply feature filtering and update solution bookkeeping.

        :param features: Feature tensor of shape (n_samples, n_features).
        :param feature_names: List of feature names.
        :return: Tuple of (filtered features, filtered feature names).
        """
        assert self.solution is not None
        if self.feature_selector is not None:
            features_filtered, filtered_names = self._apply_feature_filtering(
                features, feature_names
            )
            self.solution.set_features(features_filtered, filtered_names)
            self._feature_names = filtered_names
            features = features_filtered
            feature_names = filtered_names
        else:
            self.solution.set_features(features, feature_names)

        if self.solution.features is not None and self.solution.features.shape[0] > 0:
            n_to_show = min(10, self.solution.features.shape[1])
            if n_to_show > 0:
                logger.debug("Sample of first %d filtered features (first IC):", n_to_show)
                names_to_show = (
                    self.solution.feature_names[:n_to_show] if self.solution.feature_names else []
                )
                feature_values: list[float] = (
                    self.solution.features[0, :n_to_show].cpu().numpy().tolist()
                )
                for name, value in zip(names_to_show, feature_values, strict=False):
                    logger.debug("    %s: %.6f", name, value)

        return features, feature_names

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

    def _classify(
        self,
        features: torch.Tensor,
        unbounded_mask: torch.Tensor | None,
        n_unbounded: int,
        total_samples: int,
        original_solution: "Solution | None",
    ) -> np.ndarray:
        """Classify features and reconstruct the full label array.

        :param features: Feature tensor for bounded trajectories.
        :param unbounded_mask: Boolean mask of unbounded trajectories, or None.
        :param n_unbounded: Number of unbounded trajectories.
        :param total_samples: Total number of samples (bounded + unbounded).
        :param original_solution: Solution with the full trajectory set (pre-filtering), or None.
        :return: Label array of length total_samples.
        """
        assert self.solution is not None
        features_np = features.detach().cpu().numpy()
        bounded_labels: np.ndarray
        if is_classifier(self.predictor):
            bounded_labels = cast(SklearnClassifier, self.predictor).predict(features_np)
        else:
            bounded_labels = cast(SklearnClusterer, self.predictor).fit_predict(features_np)

        if self.detect_unbounded and unbounded_mask is not None and n_unbounded > 0:
            labels = np.empty(total_samples, dtype=object)
            labels[unbounded_mask.cpu().numpy()] = "unbounded"
            labels[~unbounded_mask.cpu().numpy()] = bounded_labels

            if original_solution is not None:
                bounded_extracted_features = self.solution.extracted_features
                bounded_extracted_feature_names = self.solution.extracted_feature_names
                bounded_features = self.solution.features
                bounded_feature_names = self.solution.feature_names
                bounded_orbit_data = self.solution.orbit_data

                self.solution = original_solution

                if bounded_extracted_features is not None:
                    self.solution.extracted_features = bounded_extracted_features
                    self.solution.extracted_feature_names = bounded_extracted_feature_names
                if bounded_features is not None:
                    self.solution.features = bounded_features
                    self.solution.feature_names = bounded_feature_names

                if bounded_orbit_data is not None:
                    bounded_mask_cpu = (~unbounded_mask).cpu()
                    self.solution.orbit_data = self._expand_orbit_data(
                        bounded_orbit_data, bounded_mask_cpu, total_samples
                    )
        else:
            labels = bounded_labels

        return labels

    def _expand_orbit_data(
        self,
        bounded_orbit_data: OrbitData,
        bounded_mask: torch.Tensor,
        total_samples: int,
    ) -> OrbitData:
        n_dof = len(bounded_orbit_data.dof_indices)
        max_peaks = bounded_orbit_data.peak_values.shape[0]
        device = bounded_orbit_data.peak_values.device
        dtype = bounded_orbit_data.peak_values.dtype

        full_peak_values = torch.full(
            (max_peaks, total_samples, n_dof), float("nan"), dtype=dtype, device=device
        )
        full_peak_counts = torch.zeros((total_samples, n_dof), dtype=torch.long, device=device)

        full_peak_values[:, bounded_mask, :] = bounded_orbit_data.peak_values
        full_peak_counts[bounded_mask, :] = bounded_orbit_data.peak_counts

        return OrbitData(
            peak_values=full_peak_values,
            peak_counts=full_peak_counts,
            dof_indices=bounded_orbit_data.dof_indices,
            time_steady=bounded_orbit_data.time_steady,
        )

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
        :raises ValueError: If ``estimate_bs()`` has not been called yet.
        """
        if self._bs_vals is None:
            raise ValueError("No results available. Please run estimate_bs() first.")

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

        :raises ValueError: If ``estimate_bs()`` has not been called yet.
        :raises ValueError: If ``output_dir`` is not defined.
        """
        if self._bs_vals is None:
            raise ValueError("No results to save. Please run estimate_bs() first.")

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

        :raises ValueError: If ``estimate_bs()`` has not been called yet.
        :raises ValueError: If ``output_dir`` is not defined.
        :raises ValueError: If no solution data is available.
        """
        if self._bs_vals is None:
            raise ValueError("No results to save. Please run estimate_bs() first.")

        if self.output_dir is None:
            raise ValueError("output_dir is not defined.")

        if self.y0 is None or self.solution is None:
            raise ValueError("No solution data available. Please run estimate_bs() first.")

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
