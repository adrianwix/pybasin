"""Accuracy tests for `PairwiseComparisonClusterer`'s algorithmic edge cases.

Validated against real `Attractors.jl` ground truth (see
`generate_algorithmic_fixtures.jl`) for behavior that isn't naturally exercised by
either full case-study pipeline (`tests/accuracy/henon_map`,
`tests/accuracy/magnetic_pendulum`): threshold-based group merging/splitting and
order-dependence of the pairwise comparison itself.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_support import julia_labels_to_pybasin

from pybasin.predictors import PairwiseComparisonClusterer

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "pairwise_algorithmic_fixtures.json"

pytestmark = pytest.mark.skipif(
    not FIXTURES_PATH.exists(),
    reason=(
        "Julia fixtures not generated; run "
        "`julia --project=packages/pybasin/tests/accuracy/_julia_project "
        "packages/pybasin/tests/accuracy/pairwise/generate_algorithmic_fixtures.jl`"
    ),
)


@pytest.fixture(scope="module")
def fixtures() -> dict[str, Any]:
    with open(FIXTURES_PATH) as f:
        return json.load(f)


class TestDumbMap:
    """`GroupViaPairwiseComparison()` defaults on an analytic 2D map."""

    def test_matches_julia_default_config(self, fixtures: dict[str, Any]) -> None:
        fx = fixtures["dumb_map"]
        X = torch.tensor(fx["features"], dtype=torch.float64)
        clusterer = PairwiseComparisonClusterer(
            threshold=fx["threshold"], rescale_features=fx["rescale_features"]
        )

        labels = np.asarray(clusterer.fit_predict(X))
        expected = julia_labels_to_pybasin(np.array(fx["labels_1based"]))
        np.testing.assert_array_equal(labels, expected)


class TestAggregationThreshold:
    """A higher threshold merges two already-identified attractors into one group."""

    def test_merge_threshold_produces_single_group(self, fixtures: dict[str, Any]) -> None:
        fx = fixtures["aggregation_threshold"]
        X = torch.tensor(fx["features"], dtype=torch.float64)
        clusterer = PairwiseComparisonClusterer(
            threshold=fx["threshold_merge"], rescale_features=False
        )

        labels = np.asarray(clusterer.fit_predict(X))
        expected = julia_labels_to_pybasin(np.array(fx["labels_merge_1based"]))
        np.testing.assert_array_equal(labels, expected)

    def test_separate_threshold_matches_unmerged_split(self, fixtures: dict[str, Any]) -> None:
        fx = fixtures["aggregation_threshold"]
        X = torch.tensor(fx["features"], dtype=torch.float64)
        clusterer = PairwiseComparisonClusterer(
            threshold=fx["threshold_separate"], rescale_features=False
        )

        labels = np.asarray(clusterer.fit_predict(X))
        expected = julia_labels_to_pybasin(np.array(fx["labels_separate_1based"]))
        np.testing.assert_array_equal(labels, expected)


class TestOrderPermutation:
    """Grouping is order-dependent: permuting the input row order can change which
    representative each point is compared against first."""

    def test_original_and_permuted_order_each_match_julia(self, fixtures: dict[str, Any]) -> None:
        fx = fixtures["order_permutation"]
        perm = np.array(fx["permutation_1based"]) - 1
        X = torch.tensor(fx["features"], dtype=torch.float64)
        clusterer = PairwiseComparisonClusterer(
            threshold=fx["threshold"], rescale_features=fx["rescale_features"]
        )

        labels_original = np.asarray(clusterer.fit_predict(X))
        expected_original = julia_labels_to_pybasin(np.array(fx["labels_original_1based"]))
        np.testing.assert_array_equal(labels_original, expected_original)

        labels_permuted = np.asarray(clusterer.fit_predict(X[perm]))
        expected_permuted = julia_labels_to_pybasin(np.array(fx["labels_permuted_1based"]))
        np.testing.assert_array_equal(labels_permuted, expected_permuted)


class TestToyChainOrder:
    """Minimal 3-point order-dependence example: [0.0, 0.9, 1.8] at threshold=1.0."""

    def test_original_and_permuted_order_each_match_julia(self, fixtures: dict[str, Any]) -> None:
        fx = fixtures["toy_chain_order"]
        perm = np.array(fx["permutation_1based"]) - 1
        X = torch.tensor(fx["values"], dtype=torch.float64)
        clusterer = PairwiseComparisonClusterer(threshold=fx["threshold"], rescale_features=False)

        labels_original = np.asarray(clusterer.fit_predict(X))
        expected_original = julia_labels_to_pybasin(np.array(fx["labels_original_1based"]))
        np.testing.assert_array_equal(labels_original, expected_original)

        labels_permuted = np.asarray(clusterer.fit_predict(X[perm]))
        expected_permuted = julia_labels_to_pybasin(np.array(fx["labels_permuted_1based"]))
        np.testing.assert_array_equal(labels_permuted, expected_permuted)
