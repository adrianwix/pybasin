from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

import numpy as np
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.sampler import Sampler
from pybasin.template_integrator import TemplateIntegrator

if TYPE_CHECKING:
    from pybasin.utils import OrbitData


class ErrorInfo(TypedDict):
    """Standard error information for basin stability estimates.

    Basin stability errors are computed using Bernoulli experiment statistics:

    - e_abs = sqrt(S_B(A) * (1 - S_B(A)) / N) - absolute standard error
    - e_rel = 1 / sqrt(N * S_B(A)) - relative standard error

    :ivar e_abs: Absolute standard error of the basin stability estimate.
    :ivar e_rel: Relative standard error of the basin stability estimate.
    """

    e_abs: float
    e_rel: float


class StudyResult(TypedDict):
    """Results for a single parameter combination in a parameter study.

    Contains complete information about basin stability estimation at one parameter
    combination, including the study label identifying the run, basin stability values,
    error estimates, sample metadata, and optional detailed solution data.

    :ivar study_label: Label identifying this run. For standalone BSE runs this is ``{"baseline": True}``;
        for study runs it maps parameter names to values (e.g. ``{"K": 0.1}``).
    :ivar basin_stability: Dictionary mapping attractor labels to their basin stability values (fraction of samples).
    :ivar errors: Dictionary mapping attractor labels to their ErrorInfo (absolute and relative errors).
    :ivar n_samples: Number of initial conditions actually used (may differ from requested N due to grid rounding).
    :ivar labels: Array of attractor labels for each initial condition, or None if not available.
    :ivar orbit_data: Peak amplitude data for orbit diagram plotting, or None if not computed.
    """

    study_label: dict[str, Any]
    basin_stability: dict[str, float]
    errors: dict[str, ErrorInfo]
    n_samples: int
    labels: np.ndarray[Any, Any] | None
    orbit_data: OrbitData | None
    initial_condition: np.ndarray[Any, Any]


class SetupProperties(TypedDict):
    """
    Standard properties returned by setup functions for case studies.

    Note: This is a flexible type definition. Actual implementations
    may use more specific types (e.g., GridSampler instead of Sampler).
    """

    n: int
    ode_system: ODESystemProtocol
    sampler: Sampler
    solver: NotRequired[SolverProtocol]
    feature_extractor: NotRequired[FeatureExtractor]
    estimator: NotRequired[BaseEstimator]
    template_integrator: NotRequired[TemplateIntegrator]
