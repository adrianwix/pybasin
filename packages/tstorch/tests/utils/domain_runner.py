"""Shared pytest functions used by each calculator-domain benchmark file."""

from __future__ import annotations

import pytest

from .benchmark_utils import Backend, FeatureSpec

BACKENDS: list[Backend] = ["tstorch_cpu", "tstorch_gpu", "tsfresh"]


def spec_id(spec: FeatureSpec) -> str:
    """Return a stable pytest parameter id for a feature spec."""
    return spec.name


def parametrize_domain(features: list[FeatureSpec]):
    """Return pytest parametrization marks for domain benchmark functions."""
    return pytest.mark.parametrize("spec", features, ids=spec_id)
