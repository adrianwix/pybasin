import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from sklearn.base import BaseEstimator, is_classifier, is_regressor  # type: ignore[import-untyped]

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
from pybasin.template_integrator import TemplateIntegrator
from pybasin.ts_torch.settings import DEFAULT_TORCH_FC_PARAMETERS
from pybasin.types import ErrorInfo
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


def _log_timing(label: str, step_time: float, total_time: float) -> None:
    if step_time > 0:
        logger.info("  %-24s %8.4fs  (%5.1f%%)", label, step_time, step_time / total_time * 100)


class BasinStabilityEstimator:
    """
    Core class for basin stability analysis.

    Configures the analysis with an ODE system, sampler, and solver,
    and provides methods to estimate basin stability and save results.

    :ivar bs_vals: Basin stability values (fraction of samples per class).
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
                      TorchDiffEqSolver for ODESystem with time_span=(0, 1000), n_steps=1000,
                      and device from sampler.
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
                time_span=(0, 1000),
                n_steps=1000,
                device=str(sampler.device),
            )
        else:
            self.solver = TorchDiffEqSolver(
                time_span=(0, 1000),
                n_steps=1000,
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

        self.bs_vals: dict[str, float] | None = None
        self.y0: torch.Tensor | None = None
        self.solution: Solution | None = None

    def _detect_unbounded_trajectories(self, y: torch.Tensor) -> torch.Tensor:
        """Detect unbounded trajectories based on Inf values.

        When JAX Diffrax integration stops due to an event, remaining timesteps are filled with Inf.

        :param y: Trajectory tensor of shape (N, B, S) where N=timesteps, B=batch, S=states.
        :return: Boolean tensor of shape (B,) indicating unbounded trajectories.
        """
        return torch.isinf(y).any(dim=(0, 2))

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

    def estimate_bs(self, parallel_integration: bool = True) -> dict[str, float]:
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
            - self.bs_vals

        :param parallel_integration: If True and using a supervised classifier with template
                                     integrator, run main and template integration in parallel.
        :return: A dictionary of basin stability values per class.
        """
        logger.info("Starting Basin Stability Estimation...")
        total_start = time.perf_counter()

        # Step 1: Sampling
        logger.info("STEP 1: Sampling Initial Conditions")
        t1 = time.perf_counter()
        self.y0 = self.sampler.sample(self.n)
        t1_elapsed = time.perf_counter() - t1
        logger.info("Generated grid with %d initial conditions in %.4fs", len(self.y0), t1_elapsed)

        # Step 2: Integration (possibly parallel with classifier fitting)
        logger.info("STEP 2: ODE Integration")
        t2_start = time.perf_counter()  # Track total integration time
        t2a_elapsed = 0.0  # Template integration time
        t2b_elapsed = 0.0  # Main integration time
        t3b_elapsed = 0.0  # Unbounded detection time
        t_orbit_elapsed = 0.0  # Orbit data extraction time
        t5b_elapsed = 0.0  # Classifier fitting time

        if parallel_integration and self.template_integrator is not None:
            logger.info("  Mode: PARALLEL (integration only)")
            logger.info("  • Main integration (sampled ICs)")
            logger.info("  • Template integration (classifier ICs)")

            with ThreadPoolExecutor(max_workers=2) as executor:
                main_future = executor.submit(self.solver.integrate, self.ode_system, self.y0)  # type: ignore[arg-type]

                template_future = executor.submit(
                    self.template_integrator.integrate,
                    self.solver,
                    self.ode_system,
                )

                t, y = main_future.result()
                template_future.result()

            t2_elapsed = time.perf_counter() - t2_start
            logger.info("Both integrations complete in %.4fs", t2_elapsed)
            logger.info("Main trajectory shape: %s", y.shape)
        else:
            if self.template_integrator is not None:
                logger.info("  Mode: SEQUENTIAL")
                logger.info("  Step 2a: Integrating template initial conditions...")
                t2a_start = time.perf_counter()
                self.template_integrator.integrate(
                    solver=self.solver,
                    ode_system=self.ode_system,
                )
                t2a_elapsed = time.perf_counter() - t2a_start
                logger.info("    Template integration in %.4fs", t2a_elapsed)

            logger.info("  Step 2b: Integrating sampled initial conditions...")
            t2b_start = time.perf_counter()
            t, y = self.solver.integrate(self.ode_system, self.y0)  # type: ignore[arg-type]
            t2b_elapsed = time.perf_counter() - t2b_start
            logger.info("    Main trajectory shape: %s", y.shape)
            logger.info("    Main integration complete in %.4fs", t2b_elapsed)

            # Total integration time includes both template and main
            t2_elapsed = time.perf_counter() - t2_start
            logger.info("    Total integration time: %.4fs", t2_elapsed)

        # Step 3: Create Solution object
        logger.info("STEP 3: Creating Solution Object")
        t3 = time.perf_counter()
        self.solution = Solution(initial_condition=self.y0, time=t, y=y)

        t3_elapsed = time.perf_counter() - t3
        logger.info("  Solution object created in %.4fs", t3_elapsed)

        # Step 3b: Detect and separate unbounded trajectories (if enabled)
        unbounded_mask: torch.Tensor | None = None
        n_unbounded = 0
        total_samples = len(self.y0)
        original_solution: Solution | None = None

        if self.detect_unbounded:
            logger.info("STEP 3b: Unboundedness Detection")
            t3b = time.perf_counter()
            unbounded_mask = self._detect_unbounded_trajectories(y)
            n_unbounded = int(unbounded_mask.sum().item())
            n_bounded = total_samples - n_unbounded
            unbounded_pct = (n_unbounded / total_samples) * 100
            t3b_elapsed = time.perf_counter() - t3b

            logger.info(
                "  Detected %d/%d unbounded trajectories (%.1f%%) in %.4fs",
                n_unbounded,
                total_samples,
                unbounded_pct,
                t3b_elapsed,
            )

            if n_unbounded == total_samples:
                logger.info(
                    "  All trajectories are unbounded. Skipping feature extraction and classification."
                )
                self.bs_vals = {"unbounded": 1.0}
                labels = np.array(["unbounded"] * total_samples, dtype=object)
                self.solution.set_labels(labels)

                total_elapsed = time.perf_counter() - total_start
                logger.info("BASIN STABILITY ESTIMATION COMPLETE")
                logger.info("Total time: %.4fs", total_elapsed)
                return self.bs_vals

            if n_unbounded > 0:
                logger.info(
                    "  Separating %d bounded trajectories for feature extraction and classification",
                    n_bounded,
                )
                bounded_mask = ~unbounded_mask

                # Store original solution for later restoration
                original_solution = self.solution

                y0_bounded = self.y0[bounded_mask]
                y_bounded = y[:, bounded_mask, :]

                self.solution = Solution(initial_condition=y0_bounded, time=t, y=y_bounded)
        else:
            logger.info("Unboundedness detection: DISABLED")

        # Extract orbit data from bounded solution (self.solution after filtering)
        if self.compute_orbit_data:
            dof = (
                list(range(self.solution.y.shape[2]))
                if self.compute_orbit_data is True
                else self.compute_orbit_data
            )
            t_orbit = time.perf_counter()
            self.solution.orbit_data = extract_orbit_data(
                self.solution.time, self.solution.y, dof=dof
            )
            t_orbit_elapsed = time.perf_counter() - t_orbit
            logger.info("Orbit data extracted in %.4fs (DOFs: %s)", t_orbit_elapsed, dof)

        # Step 4: Feature extraction (main data - fits scaler on large dataset)
        logger.info("STEP 4: Feature Extraction")
        t4 = time.perf_counter()
        features = self.feature_extractor.extract_features(self.solution)

        # Get feature names and store extracted features
        feature_names = self._get_feature_names()
        self.solution.set_extracted_features(features, feature_names)
        t4_elapsed = time.perf_counter() - t4
        logger.info("  Extracted features with shape %s in %.4fs", features.shape, t4_elapsed)

        # Step 5: Feature filtering
        logger.info("STEP 5: Feature Filtering")
        t5 = time.perf_counter()
        if self.feature_selector is not None:
            features_filtered, filtered_names = self._apply_feature_filtering(
                features, feature_names
            )
            self.solution.set_features(features_filtered, filtered_names)
            self._feature_names = filtered_names
            features = features_filtered
        else:
            self.solution.set_features(features, feature_names)
            logger.info("  No feature filtering configured")
        t5_elapsed = time.perf_counter() - t5
        logger.info("  Feature filtering complete in %.4fs", t5_elapsed)

        # Show top 10 features that remained
        if self._feature_names and len(self._feature_names) > 0:
            n_features_to_show = min(10, len(self._feature_names))
            logger.info("  Top %d features:", n_features_to_show)
            for i, feature_name in enumerate(self._feature_names[:n_features_to_show]):
                logger.info("    %d. %s", i + 1, feature_name)

        # Show sample of filtered features (first IC, up to 10 features)
        if self.solution.features is not None and self.solution.features.shape[0] > 0:
            n_features_to_show = min(10, self.solution.features.shape[1])
            if n_features_to_show > 0:
                logger.debug("Sample of first %d filtered features (first IC):", n_features_to_show)
                feature_names_filtered = (
                    self.solution.feature_names[:n_features_to_show]
                    if self.solution.feature_names
                    else []
                )
                feature_values: list[float] = (
                    self.solution.features[0, :n_features_to_show].cpu().numpy().tolist()
                )
                for name, value in zip(feature_names_filtered, feature_values, strict=False):
                    logger.debug("    %s: %.6f", name, value)

        # Step 5b: Fit classifier with template features (using already-fitted scaler)
        if self.template_integrator is not None and is_classifier(self.predictor):
            logger.info("STEP 5b: Fitting Classifier")
            t5b = time.perf_counter()
            X_train, y_train = self.template_integrator.get_training_data(
                self.feature_extractor,
                feature_selector=self.feature_selector,
            )
            cast(SklearnClassifier, self.predictor).fit(X_train, y_train)
            t5b_elapsed = time.perf_counter() - t5b
            logger.info("  Classifier fitted in %.4fs", t5b_elapsed)

        # Set feature names for predictors that require them
        final_feature_names = self._feature_names or feature_names
        if hasattr(self.predictor, "set_feature_names"):
            logger.info(
                "  Setting feature names for predictor (%d features)", len(final_feature_names)
            )
            cast(FeatureNameAware, self.predictor).set_feature_names(final_feature_names)

        # Step 6: Classification
        logger.info("STEP 6: Classification")
        t6 = time.perf_counter()

        # Convert features to numpy for predictor
        features_np = features.detach().cpu().numpy()
        bounded_labels: np.ndarray
        if is_classifier(self.predictor):
            bounded_labels = cast(SklearnClassifier, self.predictor).predict(features_np)
        else:
            bounded_labels = cast(SklearnClusterer, self.predictor).fit_predict(features_np)
        # Reconstruct full label array if unbounded trajectories were separated
        if self.detect_unbounded and unbounded_mask is not None and n_unbounded > 0:
            labels = np.empty(total_samples, dtype=object)
            labels[unbounded_mask.cpu().numpy()] = "unbounded"
            labels[~unbounded_mask.cpu().numpy()] = bounded_labels
            logger.info("  Classified %d bounded trajectories", len(bounded_labels))
            logger.info("  Reconstructed full label array with %d unbounded labels", n_unbounded)

            # Restore original solution with full trajectories
            if original_solution is not None:
                # Preserve data from bounded solution
                bounded_extracted_features = self.solution.extracted_features
                bounded_extracted_feature_names = self.solution.extracted_feature_names
                bounded_features = self.solution.features
                bounded_feature_names = self.solution.feature_names
                bounded_orbit_data = self.solution.orbit_data

                # Restore original solution
                self.solution = original_solution

                # Transfer feature data (features are computed only from bounded trajectories)
                if bounded_extracted_features is not None:
                    self.solution.extracted_features = bounded_extracted_features
                    self.solution.extracted_feature_names = bounded_extracted_feature_names
                if bounded_features is not None:
                    self.solution.features = bounded_features
                    self.solution.feature_names = bounded_feature_names

                # Expand orbit_data to full size (NaN for unbounded trajectories)
                if bounded_orbit_data is not None:
                    bounded_mask_cpu = (~unbounded_mask).cpu()
                    self.solution.orbit_data = self._expand_orbit_data(
                        bounded_orbit_data, bounded_mask_cpu, total_samples
                    )
        else:
            labels = bounded_labels

        self.solution.set_labels(labels)
        t6_elapsed = time.perf_counter() - t6
        logger.info("  Classification complete in %.4fs", t6_elapsed)

        # Step 7: Computing Basin Stability
        logger.info("STEP 7: Computing Basin Stability")
        t7 = time.perf_counter()

        # Convert all labels to strings to ensure consistent types (bounded labels may be int or str)
        labels_str = np.array([str(label) for label in labels], dtype=object)
        unique_labels, counts = np.unique(labels_str, return_counts=True)

        self.bs_vals = {str(label): 0.0 for label in unique_labels}

        # Use the actual number of samples generated, not the requested n
        # This is important because GridSampler may generate more points than requested
        actual_n = len(labels)
        fractions = counts / float(actual_n)

        for label, fraction in zip(unique_labels, fractions, strict=True):
            basin_stability_fraction = float(fraction)
            self.bs_vals[str(label)] = basin_stability_fraction
            logger.info("    %s: %.2f%%", label, basin_stability_fraction * 100)

        t7_elapsed = time.perf_counter() - t7
        logger.info("  Basin stability computed in %.4fs", t7_elapsed)

        # Summary
        total_elapsed = time.perf_counter() - total_start
        logger.info("BASIN STABILITY ESTIMATION COMPLETE")
        logger.info("Total time: %.4fs", total_elapsed)
        logger.info("Timing Breakdown:")
        _log_timing("1. Sampling:", t1_elapsed, total_elapsed)
        _log_timing("2. Integration:", t2_elapsed, total_elapsed)
        _log_timing("   - Template:", t2a_elapsed, total_elapsed)
        _log_timing("   - Main:", t2b_elapsed, total_elapsed)
        _log_timing("3. Solution/Amps:", t3_elapsed, total_elapsed)
        _log_timing("(3b) Unbounded Det.:", t3b_elapsed, total_elapsed)
        _log_timing("(3c) Orbit Data:", t_orbit_elapsed, total_elapsed)
        _log_timing("4. Features:", t4_elapsed, total_elapsed)
        _log_timing("5. Filtering:", t5_elapsed, total_elapsed)
        _log_timing("(5b) Classifier Fit:", t5b_elapsed, total_elapsed)
        _log_timing("6. Classification:", t6_elapsed, total_elapsed)
        _log_timing("7. BS Computation:", t7_elapsed, total_elapsed)

        return self.bs_vals

    def get_errors(self) -> dict[str, ErrorInfo]:
        """
        Compute absolute and relative errors for basin stability estimates.

        The errors are based on Bernoulli experiment statistics:

        - e_abs = sqrt(S_B(A) * (1 - S_B(A)) / N) — absolute standard error
        - e_rel = 1 / sqrt(N * S_B(A)) — relative error

        :return: Dictionary mapping each label to an ErrorInfo with ``e_abs`` and ``e_rel`` keys.
        :raises ValueError: If ``estimate_bs()`` has not been called yet.
        """
        if self.bs_vals is None:
            raise ValueError("No results available. Please run estimate_bs() first.")

        errors: dict[str, ErrorInfo] = {}
        n = self.n

        for label, s_b in self.bs_vals.items():
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
        if self.bs_vals is None:
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
            "basin_of_attractions": self.bs_vals,
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
        if self.bs_vals is None:
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
