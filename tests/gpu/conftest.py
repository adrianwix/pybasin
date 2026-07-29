"""Shared configuration for GPU-only, whole-pipeline tests.

These tests are deliberately separate from ``packages/pybasin/tests`` and
``tests/integration``:
they require both a CUDA device and cuML installed, run the *entire*
``BasinStabilityEstimator`` pipeline on the GPU (no forced-CPU device overrides like
the other integration tests use for reproducibility), and assert the predictor
actually resolved to the GPU backend -- not just that results look right.
"""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark every test collected under tests/gpu/ with the `gpu` marker."""
    for item in items:
        if "tests/gpu/" in str(item.path).replace("\\", "/"):
            item.add_marker(pytest.mark.gpu)
