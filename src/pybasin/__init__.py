"""pybasin: Basin stability estimation for dynamical systems."""

import logging
import sys

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter
from pybasin.plotters.types import (
    FeatureSpaceOptions,
    InteractivePlotterOptions,
    StateSpaceOptions,
    TemplatesPhaseSpaceOptions,
    TemplatesTimeSeriesOptions,
)
from pybasin.study_params import (
    CustomStudyParams,
    GridStudyParams,
    ParamAssignment,
    RunConfig,
    StudyParams,
    SweepStudyParams,
    ZipStudyParams,
)
from pybasin.types import ErrorInfo

# Configure library logger with default handler
_logger = logging.getLogger("pybasin")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

__all__ = [
    "BasinStabilityEstimator",
    "CustomStudyParams",
    "ErrorInfo",
    "FeatureSpaceOptions",
    "GridStudyParams",
    "InteractivePlotterOptions",
    "MatplotlibPlotter",
    "ParamAssignment",
    "RunConfig",
    "StateSpaceOptions",
    "StudyParams",
    "SweepStudyParams",
    "TemplatesPhaseSpaceOptions",
    "TemplatesTimeSeriesOptions",
    "ZipStudyParams",
]
