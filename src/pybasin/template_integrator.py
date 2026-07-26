"""Template integrator for supervised classifiers in basin stability analysis."""

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import FeatureSelectorProtocol, ODESystemProtocol, SolverProtocol
from pybasin.solution import Solution

logger = logging.getLogger(__name__)


class TemplateIntegrator:
    """Integrates template initial conditions and extracts training data for supervised classifiers.

    Handles the ODE integration of known attractor templates and feature extraction.

    :ivar template_y0: Template initial conditions.
    :ivar labels: Ground truth labels for each template.
    :ivar ode_params: ODE parameters for template integration (may differ from main study).
    :ivar solver: Optional dedicated solver for template integration.
    :ivar solution: Populated after :meth:`integrate` is called.
    """

    def __init__(
        self,
        template_y0: list[list[float]],
        labels: list[str],
        ode_params: Mapping[str, Any],
        solver: SolverProtocol | None = None,
    ):
        """Initialize the template integrator.

        :param template_y0: Template initial conditions as a list of lists
            (e.g., ``[[0.5, 0.0], [2.7, 0.0]]``). Will be converted to tensor
            with appropriate device during integration.
        :param labels: Ground truth labels for template conditions.
        :param ode_params: ODE parameters mapping (dict or TypedDict with numeric values).
        :param solver: Optional dedicated solver for template integration. If provided,
            this solver will be used instead of the main solver (useful for CPU-based
            template integration when templates are few).
        """
        self.template_y0 = template_y0
        self.labels = labels
        self.ode_params = deepcopy(ode_params)
        self.solver = solver
        self.solution: Solution | None = None

    @property
    def has_dedicated_solver(self) -> bool:
        """Check if a dedicated solver was provided for template integration."""
        return self.solver is not None

    def integrate(
        self,
        solver: SolverProtocol | None,
        ode_system: ODESystemProtocol,
    ) -> Solution:
        """Integrate ODE for template initial conditions.

        If no dedicated solver was provided at init, automatically integrates on
        CPU for better performance with small batch sizes (kernel-launch overhead
        dominates otherwise), then moves the resulting solution back to the main
        solver's own device. A shared feature extractor (e.g.
        ``TorchFeatureExtractorAdapter``) is locked to whichever device/shape it
        first compiles for and raises rather than silently recompiling for a
        mismatched device, so the CPU detour must stay invisible past this point.

        :param solver: Fallback solver if none was provided at init. Can be None
            if a solver was provided during initialization.
        :param ode_system: ODE system to integrate.
        :raises ValueError: If no solver is available.
        """
        template_ode_system = deepcopy(ode_system)
        template_ode_system.params = self.ode_params

        target_device = solver.device if solver is not None else None

        if self.solver is not None:
            effective_solver = self.solver
        elif solver is not None:
            if hasattr(solver, "clone") and str(solver.device) != "cpu":
                effective_solver = solver.clone(device="cpu")
                logger.debug(
                    "[TemplateIntegrator] Auto-created CPU solver (faster for small batch sizes)"
                )
            else:
                effective_solver = solver
        else:
            raise ValueError(
                "No solver available. Either pass a solver to integrate() "
                "or provide one during initialization."
            )

        logger.debug("[TemplateIntegrator] ODE params: %s", template_ode_system.params)
        logger.debug("[TemplateIntegrator] Template ICs: %d templates", len(self.template_y0))
        logger.debug("[TemplateIntegrator] Labels: %s", self.labels)
        logger.debug("[TemplateIntegrator] Using solver: %s", type(effective_solver).__name__)

        template_tensor = torch.tensor(
            self.template_y0, dtype=torch.float32, device=effective_solver.device
        )

        t, y = effective_solver.integrate(template_ode_system, template_tensor)

        if target_device is not None and str(t.device) != str(target_device):
            template_tensor = template_tensor.to(device=target_device)
            t = t.to(device=target_device)
            y = y.to(device=target_device)

        self.solution = Solution(initial_condition=template_tensor, time=t, y=y)

        return self.solution

    def get_training_data(
        self,
        feature_extractor: FeatureExtractor,
        feature_selector: FeatureSelectorProtocol | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """Extract features from integrated templates and return training data.

        Must call :meth:`integrate` first to populate ``self.solution``.

        :param feature_extractor: Feature extractor (already fitted on main data).
        :param feature_selector: Optional feature selector (already fitted on main data).
            If provided, applies the same filtering to template features.
        :param device: Target device for the returned feature tensor. Template
            batches are tiny (a handful of rows, one per label), so uploading
            them to the main study's device here is cheap and lets a
            GPU-resident classifier's ``fit()`` resolve GPU correctly. ``None``
            leaves the tensor on whatever device feature extraction produced.
        :return: Tuple of ``(X_train, y_labels)`` ready for ``classifier.fit()``.
        :raises RuntimeError: If :meth:`integrate` was not called first.
        :raises ValueError: If filtering removes all template features.
        """
        if self.solution is None:
            raise RuntimeError("Must call integrate() before get_training_data()")

        features = feature_extractor.extract_features(self.solution)

        if feature_selector is not None:
            n_original = features.shape[1]

            # Device-native selectors accept/return a tensor mirroring the input's
            # device (see FeatureSelectorProtocol); arbitrary sklearn transformers
            # are numpy-only and still need the host round trip.
            if hasattr(feature_selector, "backend"):
                features = feature_selector.transform(features)
            else:
                features_np = features.detach().cpu().numpy()
                features_filtered_np = feature_selector.transform(features_np)
                features = torch.from_numpy(features_filtered_np).to(  # type: ignore[arg-type]
                    dtype=features.dtype
                )

            if features.shape[1] == 0:
                raise ValueError(
                    f"Feature filtering removed all {n_original} template features. "
                    "This should not happen if the selector was fitted on main data."
                )

        if device is not None:
            features = features.to(device=device)

        logger.debug(
            "Template training data: %d samples, %d features",
            features.shape[0],
            features.shape[1],
        )

        return features, self.labels
