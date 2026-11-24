"""Accuracy tests for the Hénon map case study.

Validates `HistogramClusterer` and `PairwiseComparisonClusterer` against real
`Attractors.jl` ground truth (see `generate_ground_truth.jl`) through pybasin's
full pipeline: sampling, iterating the map, feature extraction, and grouping.
"""

from pathlib import Path

import numpy as np
import pytest
from case_studies.henon_map.setup_henon_map_system import setup_henon_map_system
from sklearn.base import BaseEstimator
from test_support import (
    CPU_AND_GPU_TARGETS,
    ExecutionTarget,
    compute_basin_fraction_comparisons,
    compute_classification_metrics,
    julia_labels_to_pybasin,
)

from accuracy.artifacts import AccuracyArtifactCollector, AccuracyComparison
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.predictors import HistogramClusterer, PairwiseComparisonClusterer
from pybasin.samplers import CsvSampler

GROUND_TRUTH_CSV = Path(__file__).parent / "ground_truths" / "henon_map.csv"

# Matches generate_ground_truth.jl's BIN_RANGE and
# case_studies/henon_map/main_henon_histogram_experiment.py's BIN_EDGES.
HISTOGRAM_BIN_EDGES = [-10.0, 40.0, 90.0, 140.0, 190.0, 240.0]
# Matches generate_ground_truth.jl and
# case_studies/henon_map/main_henon_pairwise_experiment.py's THRESHOLD.
PAIRWISE_THRESHOLD = 1.0


def run_henon_accuracy_test(
    predictor: BaseEstimator,
    label_column: str,
    method_name: str,
    *,
    execution_target: ExecutionTarget,
) -> AccuracyComparison:
    """Run pybasin's full Hénon map pipeline over the ground-truth initial conditions.

    :param predictor: Grouper to validate (``HistogramClusterer`` or
        ``PairwiseComparisonClusterer``).
    :param label_column: Which grouper's ground-truth column to compare against
        (``"label_histogram"`` or ``"label_pairwise"``).
    :param method_name: Stable method identifier used in documentation artifacts.
    :param execution_target: Device used by the sampler, solver, and grouper.
    :return: Complete comparison against Attractors.jl.
    """
    device = execution_target.value
    setup = setup_henon_map_system(device=device)

    sampler = CsvSampler(
        GROUND_TRUTH_CSV, coordinate_columns=["x", "y"], label_column=label_column, device=device
    )
    solver = setup.get("solver")
    assert solver is not None
    solver = solver.clone(device=device)

    bse = BasinStabilityEstimator(
        n=sampler.n_samples,
        ode_system=setup["ode_system"],
        sampler=sampler,
        solver=solver,
        feature_extractor=setup.get("feature_extractor"),
        predictor=predictor,
        feature_selector=None,
    )
    bse.run()

    assert sampler.labels is not None
    assert bse.solution is not None and bse.solution.labels is not None

    y_true = np.array([str(int(label)) for label in julia_labels_to_pybasin(sampler.labels)])
    y_pred = np.array([str(int(label)) for label in bse.solution.labels])
    return AccuracyComparison(
        system_name="henon_map",
        method_name=method_name,
        execution_target=execution_target,
        classification_metrics=compute_classification_metrics(y_true, y_pred),
        basins=compute_basin_fraction_comparisons(y_true, y_pred),
    )


@pytest.mark.parametrize("execution_target", CPU_AND_GPU_TARGETS)
class TestHenonMapHistogram:
    """`HistogramClusterer` against `GroupViaHistogram`'s labels on the Hénon map."""

    def test_baseline(
        self,
        execution_target: ExecutionTarget,
        accuracy_artifact_collector: AccuracyArtifactCollector | None,
    ) -> None:
        predictor = HistogramClusterer(bin_edges=[HISTOGRAM_BIN_EDGES, HISTOGRAM_BIN_EDGES])
        comparison = run_henon_accuracy_test(
            predictor,
            "label_histogram",
            "histogram",
            execution_target=execution_target,
        )
        metrics = comparison.classification_metrics

        assert metrics.matthews_corrcoef >= 0.99, (
            f"MCC {metrics.matthews_corrcoef:.4f} < 0.99. The Hénon map is a discrete "
            "map (no ODE integration error) with a huge bin margin around the "
            "bounded-attractor/divergence-sentinel split, so near-exact parity is expected."
        )
        assert min(metrics.f1_per_class.values()) >= 0.99
        assert all(basin.sampling_uncertainty_z_score <= 1.96 for basin in comparison.basins)
        if accuracy_artifact_collector is not None:
            accuracy_artifact_collector.add(comparison)


@pytest.mark.parametrize("execution_target", CPU_AND_GPU_TARGETS)
class TestHenonMapPairwise:
    """`PairwiseComparisonClusterer` against `GroupViaPairwiseComparison`'s labels."""

    def test_baseline(
        self,
        execution_target: ExecutionTarget,
        accuracy_artifact_collector: AccuracyArtifactCollector | None,
    ) -> None:
        predictor = PairwiseComparisonClusterer(
            threshold=PAIRWISE_THRESHOLD, rescale_features=False
        )
        comparison = run_henon_accuracy_test(
            predictor,
            "label_pairwise",
            "pairwise",
            execution_target=execution_target,
        )
        metrics = comparison.classification_metrics

        assert metrics.matthews_corrcoef >= 0.99, (
            f"MCC {metrics.matthews_corrcoef:.4f} < 0.99. The Hénon map is a discrete "
            "map (no ODE integration error) with a huge margin around the "
            "bounded-attractor/divergence-sentinel split, so near-exact parity is expected."
        )
        assert min(metrics.f1_per_class.values()) >= 0.99
        assert all(basin.sampling_uncertainty_z_score <= 1.96 for basin in comparison.basins)
        if accuracy_artifact_collector is not None:
            accuracy_artifact_collector.add(comparison)
