# pyright: reportPrivateUsage=false
"""API, kernel, and edge-case correctness tests for `PairwiseComparisonClusterer` and
`hausdorff_metric`, using hand-computed and synthetic data only.

Numerical parity against real Attractors.jl output lives in
`packages/pybasin/tests/accuracy/{magnetic_pendulum/test_magnetic_pendulum.py,
pairwise/test_algorithmic_cases.py}`.
"""

import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]

from pybasin.predictors.pairwise_comparison_clusterer import (
    PairwiseComparisonClusterer,
    _compiled_pairwise_group_scan,
    _pairwise_group_loop,
    _pairwise_group_scan,
    _rescale_columns_01,
    hausdorff_metric,
)


class TestInitIsAssignmentOnly:
    def test_defaults(self):
        c = PairwiseComparisonClusterer()
        assert c.threshold == 0.1
        assert c.metric == "minkowski"
        assert c.p == 2.0
        assert c.rescale_features is True
        assert c.initial_group_capacity == 8
        assert not hasattr(c, "resolved_params_")

    def test_clone_roundtrip(self):
        c = PairwiseComparisonClusterer(
            0.5,
            metric=hausdorff_metric,
            p=1.0,
            rescale_features=False,
            initial_group_capacity=4,
        )
        cloned = clone(c)
        assert cloned.threshold == 0.5
        assert cloned.metric is hausdorff_metric
        assert cloned.p == 1.0
        assert cloned.rescale_features is False
        assert cloned.initial_group_capacity == 4


class TestValidation:
    def test_rejects_capacity_below_one(self):
        with pytest.raises(ValueError, match="initial_group_capacity"):
            PairwiseComparisonClusterer(initial_group_capacity=0).fit_predict(np.zeros((3, 2)))

    def test_rejects_unknown_string_metric(self):
        with pytest.raises(ValueError, match="metric"):
            PairwiseComparisonClusterer(metric="euclidean").fit_predict(  # type: ignore[arg-type]
                np.zeros((3, 2))
            )

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="2D"):
            PairwiseComparisonClusterer().fit_predict(np.zeros((3, 2, 2, 2)))

    def test_rejects_rescale_with_matrix_features(self):
        X = np.zeros((3, 4, 2))
        with pytest.raises(ValueError, match="rescale_features"):
            PairwiseComparisonClusterer(metric=hausdorff_metric, rescale_features=True).fit_predict(
                X
            )


class TestEmptyAndSingletonBatches:
    def test_empty_batch_returns_empty_labels(self):
        c = PairwiseComparisonClusterer()
        labels = c.fit_predict(np.zeros((0, 3)))
        assert labels.shape == (0,)
        assert labels.dtype == np.int64
        assert c.n_groups_ == 0
        assert c.representative_indices_.shape == (0,)

    def test_singleton_batch_is_its_own_group(self):
        c = PairwiseComparisonClusterer()
        labels = c.fit_predict(np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(labels, [0])
        assert c.n_groups_ == 1


class TestExactThresholdInclusion:
    def test_equal_to_threshold_joins_existing_group(self):
        X = np.array([[0.0], [1.0]])
        c = PairwiseComparisonClusterer(threshold=1.0, rescale_features=False)
        labels = c.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 0])
        assert c.n_groups_ == 1

    def test_just_above_threshold_creates_new_group(self):
        X = np.array([[0.0], [1.0 + 1e-6]])
        c = PairwiseComparisonClusterer(threshold=1.0, rescale_features=False)
        labels = c.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1])
        assert c.n_groups_ == 2


class TestFirstRepresentativeTieBreak:
    def test_ties_pick_first_representative(self):
        # point0=-1 -> group0 (rep=-1). point1=1: dist to rep0 = 2 > 1.5 -> group1 (rep=1).
        # point2=0: equidistant (1.0) from both reps -> ties join the FIRST (group0).
        X = np.array([[-1.0], [1.0], [0.0]])
        c = PairwiseComparisonClusterer(threshold=1.5, rescale_features=False)
        labels = c.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1, 0])


class TestNearestNotFirstLeader:
    def test_joins_nearest_representative_even_when_first_is_within_threshold(self):
        # reps: 0.0 (group0) and 1.5 (group1, 1.5 > threshold from 0.0).
        # point 0.9 is within threshold of BOTH (0.9 and 0.6): it joins the nearest
        # (group1), where a first-within-threshold leader rule would pick group0.
        X = np.array([[0.0], [1.5], [0.9]])
        c = PairwiseComparisonClusterer(threshold=1.0, rescale_features=False)
        labels = c.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 1, 1])


class TestNoTransitiveClosure:
    def test_no_chaining_through_intermediate_points(self):
        # B joins A's group (dist 0.9 <= 1.0). C is far from A (dist 1.8 > 1.0) and
        # becomes a new group, even though dist(B, C) = 0.9 <= 1.0 -- C only ever compares
        # against representatives, and B never became one.
        X = np.array([[0.0], [0.9], [1.8]])
        c = PairwiseComparisonClusterer(threshold=1.0, rescale_features=False)
        labels = c.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 0, 1])


class TestOrderDependence:
    def test_permuted_order_changes_the_partition(self):
        original = np.array([[0.0], [0.9], [1.8]])
        permuted = original[[2, 1, 0]]

        labels_original = PairwiseComparisonClusterer(
            threshold=1.0, rescale_features=False
        ).fit_predict(original)
        labels_permuted = PairwiseComparisonClusterer(
            threshold=1.0, rescale_features=False
        ).fit_predict(permuted)

        # original: 0.0 and 0.9 share a group, 1.8 is alone.
        assert labels_original[0] == labels_original[1] != labels_original[2]
        # permuted (mapped back to original indices [2, 1, 0]): 0.9 and 1.8 share a
        # group instead, 0.0 is alone -- a different partition, not just different labels.
        remapped = np.empty(3, dtype=labels_permuted.dtype)
        remapped[[2, 1, 0]] = labels_permuted
        assert remapped[1] == remapped[2] != remapped[0]


class TestRescaling:
    def test_rescale_maps_columns_to_01(self):
        X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        c = PairwiseComparisonClusterer(rescale_features=True)
        c.fit_predict(X)
        np.testing.assert_allclose(c.rescale_min_, [0.0, 10.0])
        np.testing.assert_allclose(c.rescale_range_, [10.0, 20.0])

    def test_constant_column_becomes_zero_not_nan(self):
        X = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaled, _column_min, column_range = _rescale_columns_01(X)
        assert torch.equal(scaled[:, 0], torch.zeros(3, dtype=scaled.dtype))
        assert not torch.isnan(scaled).any()
        assert column_range[0] == 1.0

    def test_constant_column_does_not_collapse_groups(self):
        # Column 0 is constant; column 1 separates two clouds after rescaling to [0, 1].
        X = np.array([[5.0, 0.0], [5.0, 0.01], [5.0, 10.0], [5.0, 10.01]])
        c = PairwiseComparisonClusterer(threshold=0.1, rescale_features=True)
        labels = c.fit_predict(X)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])

    def test_rescale_false_does_not_set_rescale_attrs(self):
        c = PairwiseComparisonClusterer(rescale_features=False)
        c.fit_predict(np.array([[0.0], [1.0]]))
        assert not hasattr(c, "rescale_min_")
        assert not hasattr(c, "rescale_range_")


class TestCapacityRetryInvariance:
    @staticmethod
    def _three_cluster_batch() -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.concatenate([
            rng.normal(loc=[0, 0], scale=0.05, size=(20, 2)),
            rng.normal(loc=[5, 5], scale=0.05, size=(20, 2)),
            rng.normal(loc=[-5, 5], scale=0.05, size=(20, 2)),
        ])

    @pytest.mark.parametrize("capacity", [1, 2, 8])
    def test_capacity_does_not_change_labels(self, capacity: int):
        X = self._three_cluster_batch()
        baseline = PairwiseComparisonClusterer(
            threshold=1.0, rescale_features=False, initial_group_capacity=8
        ).fit_predict(X)
        labels = PairwiseComparisonClusterer(
            threshold=1.0, rescale_features=False, initial_group_capacity=capacity
        ).fit_predict(X)
        np.testing.assert_array_equal(labels, baseline)


class TestRepeatedCallStateReplacement:
    def test_second_call_replaces_first_call_state(self):
        c = PairwiseComparisonClusterer(threshold=1.0, rescale_features=False)
        c.fit_predict(np.array([[0.0], [0.0], [0.0]]))
        assert c.n_groups_ == 1

        c.fit_predict(np.array([[0.0], [5.0], [10.0]]))
        assert c.n_groups_ == 3
        assert c.representative_indices_.shape == (3,)


class TestHausdorffMetric:
    def test_matches_hand_computed_example(self):
        a = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        b = torch.tensor([[0.0, 0.0]])
        # Every point in a is within {0, 1} of b's single point (max=1); b's point is
        # within {0} of its nearest point in a (max=0). Hausdorff = max(1, 0) = 1.
        assert float(hausdorff_metric(a, b)) == pytest.approx(1.0)

    def test_zero_for_identical_sets(self):
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert float(hausdorff_metric(a, a)) == pytest.approx(0.0)

    def test_symmetric(self):
        a = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        b = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        assert float(hausdorff_metric(a, b)) == pytest.approx(float(hausdorff_metric(b, a)))


class TestKernelValidationGate:
    """The primary `scan` kernel's validation gate against synthetic data:
    1. eager `scan` matches the plain per-row reference loop exactly.
    2. compiled `scan` matches eager `scan` exactly.
    Both `torch.while_loop` and `torch._higher_order_ops.scan` are labeled prototype by
    PyTorch itself ("you may run into miscompiles") -- this is the actual check that gate
    passed for the pinned torch version, not a formality.
    """

    @staticmethod
    def _vector_batch(dtype: torch.dtype, n: int = 40, seed: int = 0) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        quarter = n // 4
        X = np.concatenate([
            rng.normal(loc=[0, 0, 0], scale=0.1, size=(quarter, 3)),
            rng.normal(loc=[3, 0, 0], scale=0.1, size=(quarter, 3)),
            rng.normal(loc=[0, 3, 0], scale=0.1, size=(quarter, 3)),
            rng.normal(loc=[0, 0, 3], scale=0.1, size=(n - 3 * quarter, 3)),
        ])
        return torch.tensor(X, dtype=dtype)

    @staticmethod
    def _matrix_batch(dtype: torch.dtype, n: int = 12, m: int = 4, seed: int = 2) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        centers = [np.array([0.0, 0.0]), np.array([5.0, 0.0]), np.array([0.0, 5.0])]
        samples = [centers[i % 3] + rng.normal(scale=0.1, size=(m, 2)) for i in range(n)]
        return torch.tensor(np.stack(samples), dtype=dtype)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("capacity", [1, 2, 8, 40])
    def test_eager_scan_matches_reference_loop_minkowski(self, dtype: torch.dtype, capacity: int):
        X = self._vector_batch(dtype)
        idx = torch.arange(X.shape[0], dtype=torch.int64)
        threshold = torch.as_tensor(1.0, dtype=dtype)

        labels_scan, rep_idx_scan, n_groups_scan, overflowed = _pairwise_group_scan(
            X, idx, threshold, capacity, 2.0, True, None
        )
        labels_ref, rep_idx_ref, n_groups_ref = _pairwise_group_loop(
            X, 1.0, use_cdist=True, p=2.0, metric_fn=None
        )

        if capacity >= n_groups_ref:
            assert not bool(overflowed)
            assert int(n_groups_scan) == n_groups_ref
            torch.testing.assert_close(labels_scan, labels_ref)
            torch.testing.assert_close(rep_idx_scan[:n_groups_ref], rep_idx_ref)
        else:
            assert bool(overflowed)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("capacity", [1, 2, 8])
    def test_eager_scan_matches_reference_loop_hausdorff(self, dtype: torch.dtype, capacity: int):
        X = self._matrix_batch(dtype)
        idx = torch.arange(X.shape[0], dtype=torch.int64)
        threshold = torch.as_tensor(1.0, dtype=dtype)

        labels_scan, _rep_idx_scan, n_groups_scan, overflowed = _pairwise_group_scan(
            X, idx, threshold, capacity, 2.0, False, hausdorff_metric
        )
        labels_ref, _rep_idx_ref, n_groups_ref = _pairwise_group_loop(
            X, 1.0, use_cdist=False, p=2.0, metric_fn=hausdorff_metric
        )

        if capacity >= n_groups_ref:
            assert not bool(overflowed)
            assert int(n_groups_scan) == n_groups_ref
            torch.testing.assert_close(labels_scan, labels_ref)
        else:
            assert bool(overflowed)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_compiled_matches_eager_minkowski(self, dtype: torch.dtype):
        X = self._vector_batch(dtype)
        idx = torch.arange(X.shape[0], dtype=torch.int64)
        threshold = torch.as_tensor(1.0, dtype=dtype)
        capacity = 8

        eager = _pairwise_group_scan(X, idx, threshold, capacity, 2.0, True, None)
        compiled = _compiled_pairwise_group_scan(X, idx, threshold, capacity, 2.0, True, None)
        for e, c in zip(eager, compiled, strict=True):
            torch.testing.assert_close(e, c)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_compiled_matches_eager_hausdorff(self, dtype: torch.dtype):
        X = self._matrix_batch(dtype)
        idx = torch.arange(X.shape[0], dtype=torch.int64)
        threshold = torch.as_tensor(1.0, dtype=dtype)
        capacity = 8

        eager = _pairwise_group_scan(X, idx, threshold, capacity, 2.0, False, hausdorff_metric)
        compiled = _compiled_pairwise_group_scan(
            X, idx, threshold, capacity, 2.0, False, hausdorff_metric
        )
        for e, c in zip(eager, compiled, strict=True):
            torch.testing.assert_close(e, c)
