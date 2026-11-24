"""Execution targets and assertions for CPU/GPU accuracy tests."""

import importlib.util
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA GPU",
)
requires_gpu_and_cuml = pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("cuml") is not None),
    reason="requires a CUDA GPU with cuML installed",
)


class ExecutionTarget(Enum):
    """Device target for an accuracy-test pipeline."""

    CPU = "cpu"
    GPU = "cuda"

    @property
    def is_gpu(self) -> bool:
        """Whether this target exercises the CUDA pipeline."""
        return self is ExecutionTarget.GPU


CPU_AND_GPU_TARGETS = (
    pytest.param(ExecutionTarget.CPU, id="cpu"),
    pytest.param(
        ExecutionTarget.GPU,
        marks=[pytest.mark.gpu, requires_cuda],
        id="gpu",
    ),
)

CPU_AND_GPU_WITH_CUML_TARGETS = (
    pytest.param(ExecutionTarget.CPU, id="cpu"),
    pytest.param(
        ExecutionTarget.GPU,
        marks=[pytest.mark.gpu, requires_gpu_and_cuml],
        id="gpu",
    ),
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
    results end to end, not a byte-for-byte regression check.
    """
    mismatches: list[str] = []
    for label, expected_value in expected.items():
        if expected_value == 0.0:
            continue
        actual_value = actual.get(label, 0.0)
        if abs(actual_value - expected_value) > atol:
            mismatches.append(f"{label}: expected {expected_value:.4f}, got {actual_value:.4f}")
    assert not mismatches, "Basin stability values not near ground truth:\n" + "\n".join(mismatches)


def assert_predictor_used_backend(predictor: Any, expected: Literal["cpu", "gpu"]) -> None:
    """Assert that a fitted backend-neutral predictor used the expected backend.

    Works for both a bare backend-neutral predictor (``backend_`` set directly) and a
    ``TunedClusterer`` wrapping one (``backend_`` set on the fitted inner ``estimator_``).
    """
    backend_ = getattr(predictor, "backend_", None)
    if backend_ is None:
        inner = getattr(predictor, "estimator_", None)
        backend_ = getattr(inner, "backend_", None)
    assert backend_ == expected, (
        f"Expected predictor to resolve the {expected!r} backend, got backend_={backend_!r}."
    )


def assert_predictor_used_gpu(predictor: Any) -> None:
    """Assert that a fitted backend-neutral predictor used its GPU backend."""
    assert_predictor_used_backend(predictor, "gpu")
