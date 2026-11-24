"""API, kernel, and edge-case correctness tests for `HistogramClusterer`.

Numerical parity against real Attractors.jl output lives in
`packages/pybasin/tests/accuracy/henon_map/test_henon_map.py`.
"""

from typing import Literal

import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]

from pybasin.predictors.histogram_clusterer import (
    HistogramClusterer,
    _strides_from_n_bins,  # pyright: ignore[reportPrivateUsage]
    _validate_label_capacity,  # pyright: ignore[reportPrivateUsage]
)


class TestInitIsAssignmentOnly:
    def test_defaults(self):
        grouper = HistogramClusterer(bin_edges=[[0, 1, 2]])
        assert grouper.bin_edges == [[0, 1, 2]]
        assert grouper.feature_names is None
        assert grouper.boundary_mode == "searchsorted"
        assert not hasattr(grouper, "resolved_params_")

    def test_clone_roundtrip(self):
        grouper = HistogramClusterer(
            bin_edges=[[0, 1, 2], [-1, 0, 1]],
            feature_names=["a", "b"],
            boundary_mode="arithmetic",
        )
        cloned = clone(grouper)
        assert cloned.bin_edges == [[0, 1, 2], [-1, 0, 1]]
        assert cloned.feature_names == ["a", "b"]
        assert cloned.boundary_mode == "arithmetic"
        assert cloned is not grouper


class TestKernelBasics:
    """1D/2D grids with exact, hand-worked expected bin indices."""

    def test_1d_interior_points(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0, 3.0]])
        X = np.array([[0.5], [1.5], [2.5]])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_1d_internal_edge_belongs_to_right_bin(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0, 3.0]])
        X = np.array([[0.0], [1.0], [2.0]])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1, 2])

    def test_1d_final_edge_is_out_of_range(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0, 3.0]])
        X = np.array([[3.0]])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [-1])

    def test_1d_below_and_above_range(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0, 3.0]])
        X = np.array([[-0.5], [3.5]])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [-1, -1])

    def test_2d_first_axis_fastest_strides(self):
        # 2 bins on axis 0, 3 bins on axis 1 -> strides = [1, 2]
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0, 3.0]])
        # cartesian (row, col) 0-based -> linear = row + col * 2
        X = np.array([
            [0.5, 0.5],  # (0, 0) -> 0
            [1.5, 0.5],  # (1, 0) -> 1
            [0.5, 1.5],  # (0, 1) -> 2
            [1.5, 1.5],  # (1, 1) -> 3
            [0.5, 2.5],  # (0, 2) -> 4
            [1.5, 2.5],  # (1, 2) -> 5
        ])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1, 2, 3, 4, 5])
        assert grouper.n_bins_ == (2, 3)
        assert grouper.strides_ == (1, 2)

    def test_three_feature_dimensions(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        # strides = [1, 2, 4]
        X = np.array([
            [0.5, 0.5, 0.5],  # (0,0,0) -> 0
            [1.5, 0.5, 0.5],  # (1,0,0) -> 1
            [0.5, 1.5, 0.5],  # (0,1,0) -> 2
            [0.5, 0.5, 1.5],  # (0,0,1) -> 4
            [1.5, 1.5, 1.5],  # (1,1,1) -> 7
        ])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1, 2, 4, 7])
        assert grouper.n_bins_ == (2, 2, 2)
        assert grouper.strides_ == (1, 2, 4)

    def test_sparse_occupied_bins_keep_original_linear_ids(self):
        # A grid with 10 bins per axis but only two occupied, far apart -- labels
        # must be the true linear ids, not a 0..k-1 relabeling of occupied bins.
        edges = list(range(11))  # 10 bins, edges 0..10
        grouper = HistogramClusterer(bin_edges=[edges, edges])
        X = np.array([[0.5, 0.5], [9.5, 9.5]])
        labels = grouper.fit_predict(X)
        # (0,0) -> 0 ; (9,9) -> 9 + 9*10 = 99
        np.testing.assert_array_equal(labels, [0, 99])


class TestBoundaryModes:
    @pytest.mark.parametrize("boundary_mode", ["searchsorted", "arithmetic"])
    def test_uniform_edges_agree_on_interior_and_edge_points(
        self, boundary_mode: Literal["searchsorted", "arithmetic"]
    ):
        grouper = HistogramClusterer(
            bin_edges=[[0.0, 1.0, 2.0, 3.0, 4.0]], boundary_mode=boundary_mode
        )
        X = np.array([[0.0], [0.5], [1.0], [2.999], [4.0], [-1.0], [5.0]])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 0, 1, 2, -1, -1, -1])

    @pytest.mark.parametrize("boundary_mode", ["searchsorted", "arithmetic"])
    @pytest.mark.parametrize(("lower", "upper", "n_bins"), [(-2.0, 2.0, 7), (-3.3, 9.1, 13)])
    def test_value_just_below_upper_edge_never_aliases_another_bin(
        self,
        boundary_mode: Literal["searchsorted", "arithmetic"],
        lower: float,
        upper: float,
        n_bins: int,
    ):
        # For these edges, floor((x - lower) / width) rounds to n_bins at
        # x = nextafter(upper, -inf). That index must not spill into the second
        # axis's stride and produce the label of a different bin.
        edges = np.linspace(lower, upper, n_bins + 1)
        x = np.nextafter(upper, -np.inf)
        grouper = HistogramClusterer(bin_edges=[edges, edges], boundary_mode=boundary_mode)
        labels = grouper.fit_predict(np.array([[x, lower]]))
        assert labels[0] in (n_bins - 1, -1)

    def test_arithmetic_rejects_non_uniform_edges(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 3.0, 4.0]], boundary_mode="arithmetic")
        with pytest.raises(ValueError, match="equally spaced"):
            grouper.fit_predict(np.array([[0.5]]))

    def test_searchsorted_allows_non_uniform_edges(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 3.0, 4.0]], boundary_mode="searchsorted")
        X = np.array([[0.5], [2.0], [3.5]])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1, 2])


class TestEdgeCases:
    def test_nan_gives_minus_one_for_whole_row(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        X = np.array([[float("nan"), 0.5, 0.5]])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [-1])

    def test_inf_gives_minus_one_for_whole_row(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        X = np.array([
            [float("inf"), 0.5, 0.5],
            [0.5, float("-inf"), 0.5],
        ])
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [-1, -1])

    def test_empty_input_returns_empty_int64_labels(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0]])
        X = np.empty((0, 1))
        labels = grouper.fit_predict(X)
        assert labels.shape == (0,)
        assert labels.dtype == np.int64

    def test_wrong_feature_count_raises(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="feature columns"):
            grouper.fit_predict(np.array([[0.5]]))

    def test_duplicate_edges_raise(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 1.0, 2.0]])
        with pytest.raises(ValueError, match="strictly ascending"):
            grouper.fit_predict(np.array([[0.5]]))

    def test_descending_edges_raise(self):
        grouper = HistogramClusterer(bin_edges=[[2.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="strictly ascending"):
            grouper.fit_predict(np.array([[0.5]]))

    def test_non_finite_edges_raise(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, float("inf")]])
        with pytest.raises(ValueError, match="finite"):
            grouper.fit_predict(np.array([[0.5]]))

    def test_too_few_edges_raise(self):
        grouper = HistogramClusterer(bin_edges=[[0.0]])
        with pytest.raises(ValueError, match="at least 2 entries"):
            grouper.fit_predict(np.array([[0.5]]))

    def test_mixed_integer_and_float_edges_promote_to_floating(self):
        # axis 0: python ints: axis 1: python floats
        grouper = HistogramClusterer(bin_edges=[[0, 1, 2], [0.0, 1.0, 2.0]])
        X = np.array([[0.5, 0.5]], dtype=np.float32)
        labels = grouper.fit_predict(X)
        np.testing.assert_array_equal(labels, [0])
        assert labels.dtype == np.int64

    def test_label_capacity_validation_rejects_overflow(self):
        with pytest.raises(ValueError, match="does not fit a signed 64-bit label"):
            _validate_label_capacity([2**32, 2**32, 4])

    def test_label_capacity_validation_accepts_normal_sizes(self):
        _validate_label_capacity([100, 100, 100])  # should not raise

    def test_strides_first_axis_fastest(self):
        assert _strides_from_n_bins([3]) == [1]
        assert _strides_from_n_bins([3, 4]) == [1, 3]
        assert _strides_from_n_bins([3, 4, 5]) == [1, 3, 12]


class TestFeatureIdentity:
    def test_no_feature_names_configured_accepts_anything(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0]])
        grouper.set_feature_names(["whatever", "order"])  # no error

    def test_exact_match_accepted(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0], [0.0, 1.0]], feature_names=["a", "b"])
        grouper.set_feature_names(["a", "b"])  # no error

    def test_reordered_columns_rejected(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0], [0.0, 1.0]], feature_names=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            grouper.set_feature_names(["b", "a"])

    def test_dropped_column_rejected(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0], [0.0, 1.0]], feature_names=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            grouper.set_feature_names(["a"])


class TestDtypeDeviceMirroring:
    def test_numpy_in_numpy_out(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0]])
        X = np.array([[0.5]])
        labels = grouper.fit_predict(X)
        assert isinstance(labels, np.ndarray)

    def test_torch_cpu_in_torch_cpu_out(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0]])
        X = torch.tensor([[0.5]])
        labels = grouper.fit_predict(X)
        assert isinstance(labels, torch.Tensor)
        assert labels.device.type == "cpu"
        assert labels.dtype == torch.int64

    def test_torch_float64_in_still_int64_labels(self):
        grouper = HistogramClusterer(bin_edges=[[0.0, 1.0, 2.0]])
        X = torch.tensor([[0.5]], dtype=torch.float64)
        labels = grouper.fit_predict(X)
        assert labels.dtype == torch.int64


class TestDiagnostics:
    def test_resolved_params_reflects_construction(self):
        grouper = HistogramClusterer(
            bin_edges=[[0.0, 1.0, 2.0]], feature_names=["a"], boundary_mode="arithmetic"
        )
        grouper.fit_predict(np.array([[0.5]]))
        assert grouper.resolved_params_["bin_edges"] == [[0.0, 1.0, 2.0]]
        assert grouper.resolved_params_["feature_names"] == ["a"]
        assert grouper.resolved_params_["boundary_mode"] == "arithmetic"
