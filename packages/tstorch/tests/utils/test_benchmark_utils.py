# pyright: basic
"""Tests for `benchmark_utils.py`.

Uses the synthetic `Scale` feature so these checks exercise the benchmark helper behavior,
not a real calculator's implementation.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from tstorch import torch_feature_extractor

from .benchmark_utils import (
    run_feature_class_benchmark,
    run_single_feature_extractor_proxy_call_benchmark,
    tsfresh_argument_batching,
)
from .feature_specs import DOMAIN_FEATURES
from .synthetic_feature import Scale, ScaleParams

ARGUMENT_COUNTS = [1, 5, 10]
AUTOCORRELATION_FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["autocorrelation"]}


@pytest.fixture
def registered_scale(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(torch_feature_extractor.SUPPORTED_FEATURES, "scale", Scale)


class TestRunFeatureClassBenchmark:
    @pytest.mark.parametrize("count", ARGUMENT_COUNTS)
    def test_scales_with_argument_count(self, benchmark, count: int) -> None:
        y_bst = torch.randn(10_000, 1, 50)
        params = [{"scale": i} for i in range(1, count + 1)]

        result = run_feature_class_benchmark(benchmark, y_bst, Scale, params, device="cpu")

        assert result.shape == (10_000, 1, count)

    def test_matches_direct_construction(self, benchmark) -> None:
        y_bst = torch.randn(100, 2, 20)
        params = [{"scale": 2}, {"scale": 3}]

        result = run_feature_class_benchmark(benchmark, y_bst, Scale, params, device="cpu")

        torch.testing.assert_close(result, Scale(cast(list[ScaleParams], params))(y_bst))


class TestRunSingleFeatureExtractorProxyCallBenchmark:
    @pytest.mark.usefixtures("registered_scale")
    @pytest.mark.parametrize("use_compile", [False, True])
    def test_runs_at_single_feature_scale(self, benchmark, use_compile: bool) -> None:
        y_bst = torch.randn(3, 2, 50)

        result = run_single_feature_extractor_proxy_call_benchmark(
            benchmark,
            y_bst,
            "scale",
            [{"scale": 2}],
            device="cpu",
            use_compile=use_compile,
        )

        assert result.shape == (3, 2)

    @pytest.mark.usefixtures("registered_scale")
    def test_compiled_warmup_call_is_untimed(self, benchmark) -> None:
        y_bst = torch.randn(2, 1, 20)

        first = run_single_feature_extractor_proxy_call_benchmark(
            benchmark, y_bst, "scale", [{"scale": 3}], device="cpu", use_compile=True
        )

        torch.testing.assert_close(first, Scale([{"scale": 3}])(y_bst)[:, 0, :])


class TestTsfreshArgumentBatching:
    pytestmark = pytest.mark.usefixtures("_tsfresh_available")

    def test_simple_calculator_loops_over_arguments(self) -> None:
        spec = AUTOCORRELATION_FEATURES["autocorrelation"]

        assert tsfresh_argument_batching(spec) == "proxy_loops_per_argument"
        assert "feature_calculators.autocorrelation" in spec.tsfresh_argument_batching_source

    @pytest.mark.parametrize("feature_name", ["partial_autocorrelation", "agg_autocorrelation"])
    def test_combiner_calculator_gets_parameter_list(self, feature_name: str) -> None:
        spec = AUTOCORRELATION_FEATURES[feature_name]

        assert tsfresh_argument_batching(spec) == "calculator_gets_param_list_and_shares_work"
        assert f"feature_calculators.{feature_name}" in spec.tsfresh_argument_batching_source


@pytest.fixture
def _tsfresh_available():
    pytest.importorskip("tsfresh")
