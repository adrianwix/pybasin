# pyright: basic
"""Tests for `unit_test_utils.py`."""

from __future__ import annotations

from typing import TypedDict

import pytest
import torch
from tstorch.feature_factory import Feature, ParameterlessFeature

from . import unit_test_utils
from .benchmark_utils import BenchmarkSeriesData, FeatureSpec
from .synthetic_feature import Scale
from .unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
)


@pytest.fixture
def y_bst() -> torch.Tensor:
    return torch.randn(2, 3, 10)


@pytest.fixture
def canonical_trajectories() -> BenchmarkSeriesData:
    first = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]])
    second = torch.tensor([[[3.0, 6.0]], [[4.0, 8.0]]])
    return BenchmarkSeriesData(
        time=torch.arange(3),
        y_bst=first,
        case_tensors={"first": first, "second": second},
        case_times={},
        case_slices={},
        metadata={},
    )


class _Mean(ParameterlessFeature):
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1, keepdim=True)

    @property
    def names(self) -> list[str]:
        return ["mean"]


class _OffsetParams(TypedDict):
    offset: float


class _Offset(Feature[_OffsetParams]):
    def __init__(self, params: list[_OffsetParams]) -> None:
        self.params = params

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        offsets = torch.tensor(
            [float(param["offset"]) for param in self.params], dtype=x.dtype, device=x.device
        )
        return mean + offsets

    @property
    def names(self) -> list[str]:
        return [f"offset__{param['offset']}" for param in self.params]


class _BrokenParams(TypedDict):
    pass


class _Broken(Feature[_BrokenParams]):
    def __init__(self, params: list[_BrokenParams]) -> None:
        self.params = params

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean().item() > 0:  # data-dependent Python branch -> untraceable
            return x.sum(dim=-1, keepdim=True)
        return x.sum(dim=-1, keepdim=True) * 2

    @property
    def names(self) -> list[str]:
        return [""]


def _flat_means(y_bst: torch.Tensor) -> list[float]:
    return y_bst.mean(dim=-1).reshape(-1).tolist()


class TestAssertFeatureMatchesTsfresh:
    def test_parameterless_records_all_canonical_cases(
        self, monkeypatch: pytest.MonkeyPatch, canonical_trajectories: BenchmarkSeriesData
    ) -> None:
        records: list[dict[str, object]] = []
        monkeypatch.setattr(
            unit_test_utils,
            "execute_tsfresh_direct",
            lambda y_bst, spec: _flat_means(y_bst),
        )
        monkeypatch.setattr(
            unit_test_utils,
            "record_precision",
            lambda feature, params, **stats: records.append(stats),
        )

        assert_feature_matches_tsfresh(
            _Mean, FeatureSpec("mean", "synthetic"), canonical_trajectories
        )

        assert records == [
            {
                "dtype": "float32",
                "rows_compared": 4,
                "total_abs_diff": 0.0,
                "max_abs_diff": 0.0,
                "nan_mismatches": 0,
                "passed": True,
                "atol": 1e-3,
                "rtol": 1e-3,
            }
        ]

    @pytest.mark.parametrize("fctype", ["simple", "combiner"])
    def test_parametrized_records_every_param_for_all_canonical_cases(
        self,
        monkeypatch: pytest.MonkeyPatch,
        canonical_trajectories: BenchmarkSeriesData,
        fctype: str,
    ) -> None:
        params = [{"offset": 1.0}, {"offset": 2.0}]
        records: list[tuple[dict[str, object] | None, dict[str, object]]] = []

        def execute(y_bst: torch.Tensor, spec: FeatureSpec) -> list[object]:
            means = _flat_means(y_bst)
            if spec.tsfresh_param is not None:
                return [
                    zip(
                        [""] * len(spec.tsfresh_param),
                        [mean + float(param["offset"]) for param in spec.tsfresh_param],
                        strict=True,
                    )
                    for mean in means
                ]
            offset = float((spec.tsfresh_kwargs or {})["offset"])
            return [mean + offset for mean in means]

        monkeypatch.setattr(unit_test_utils, "execute_tsfresh_direct", execute)
        monkeypatch.setattr(unit_test_utils, "tsfresh_dispatcher_fctype", lambda spec: fctype)
        monkeypatch.setattr(
            unit_test_utils,
            "record_precision",
            lambda feature, report_params, **stats: records.append((report_params, stats)),
        )

        assert_feature_matches_tsfresh(
            _Offset,
            FeatureSpec("offset", "synthetic"),
            canonical_trajectories,
            params,
        )

        assert [record_params for record_params, _ in records] == params
        assert [stats["rows_compared"] for _, stats in records] == [4, 4]
        assert all(stats["passed"] for _, stats in records)


class TestAssertFeatureCompilesFullgraph:
    def test_scale_compiles_fullgraph(self, y_bst) -> None:
        assert_feature_compiles_fullgraph("scale", Scale, [{"scale": 2}], y_bst)

    def test_data_dependent_branch_fails_with_report(self, y_bst) -> None:
        with pytest.raises(AssertionError, match="is not torch.compile\\(fullgraph=True\\)-safe"):
            assert_feature_compiles_fullgraph("broken", _Broken, [{}], y_bst)


class TestAssertFeatureExportsCleanly:
    def test_scale_exports_cleanly(self, y_bst) -> None:
        assert_feature_exports_cleanly("scale", Scale, [{"scale": 2}], y_bst)

    def test_data_dependent_branch_fails_with_report(self, y_bst) -> None:
        with pytest.raises(AssertionError, match="does not torch.export cleanly"):
            assert_feature_exports_cleanly("broken", _Broken, [{}], y_bst)
