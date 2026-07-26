"""Helpers for whole-pipeline GPU tests.

Unlike ``tests/integration/test_helpers.py``, these deliberately do *not* force
CPU anywhere -- the point is to prove the entire pipeline (sampling, integration,
feature extraction, clustering/classification) runs on the GPU end to end, using
the real cuML backend. Comparisons against ground truth use a loose tolerance;
exact classification-metric parity is not the goal here (that's already covered
on CPU by ``tests/integration``).
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch

requires_gpu_and_cuml = pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("cuml") is not None),
    reason="requires a CUDA GPU with cuML installed",
)


def load_expected_basin_stability(json_path: Path) -> dict[str, float]:
    """Load {display_label: basinStability} from a `main_*_supervised.json` reference file."""
    with open(json_path) as f:
        meta = json.load(f)
    label_map: dict[str, str] = meta.get("label_map", {})
    expected: dict[str, float] = {}
    for entry in meta["data"]:
        label = str(entry["label"])
        display_label = label_map.get(label, label)
        expected[display_label] = float(entry["basinStability"])
    return expected


def assert_basin_stability_near(
    actual: dict[str, float],
    expected: dict[str, float],
    *,
    atol: float = 0.05,
) -> None:
    """Assert every non-zero expected label's basin stability is within `atol` of actual.

    Loose by design: this is evidence the GPU pipeline produces sane, roughly-correct
    results end to end, not a byte-for-byte regression check (that's what the CPU-forced
    tests in tests/integration are for).
    """
    mismatches: list[str] = []
    for label, expected_value in expected.items():
        if expected_value == 0.0:
            continue
        actual_value = actual.get(label, 0.0)
        if abs(actual_value - expected_value) > atol:
            mismatches.append(f"{label}: expected {expected_value:.4f}, got {actual_value:.4f}")
    assert not mismatches, "Basin stability values not near ground truth:\n" + "\n".join(mismatches)


def assert_predictor_used_gpu(predictor: Any) -> None:
    """Assert a fitted predictor actually resolved to the cuML/GPU backend, not a silent fallback.

    Works for both a bare backend-neutral predictor (``backend_`` set directly) and a
    ``TunedClusterer`` wrapping one (``backend_`` set on the fitted inner ``estimator_``).
    """
    backend_ = getattr(predictor, "backend_", None)
    if backend_ is None:
        inner = getattr(predictor, "estimator_", None)
        backend_ = getattr(inner, "backend_", None)
    assert backend_ == "gpu", (
        f"Expected predictor to resolve the GPU backend, got backend_={backend_!r}. "
        "This means the pipeline silently fell back to CPU/sklearn instead of cuML."
    )
