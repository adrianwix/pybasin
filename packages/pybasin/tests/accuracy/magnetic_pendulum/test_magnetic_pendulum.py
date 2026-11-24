"""Accuracy tests for the magnetic pendulum case study.

Validates `HistogramClusterer` and `PairwiseComparisonClusterer` against real
`Attractors.jl` ground truth (see `generate_ground_truth.jl`) through pybasin's
full pipeline: sampling, integrating the ODE, feature extraction, and grouping.

The pendulum's basins have fractal boundaries, so a small fraction of points near a
boundary can legitimately land in a different group between pybasin's ODE solver and
Attractors.jl's Vern9 integration without indicating a bug -- MCC thresholds below are
looser than the Hénon map's discrete-map suite for this reason.
"""

from pathlib import Path

import numpy as np
import pytest
from case_studies.magnetic_pendulum.setup_magnetic_pendulum_system import (
    setup_magnetic_pendulum_system,
)
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

GROUND_TRUTH_CSV = Path(__file__).parent / "ground_truths" / "magnetic_pendulum.csv"

# Matches generate_ground_truth.jl's BIN_RANGE_X/BIN_RANGE_Y and
# case_studies/magnetic_pendulum/main_magnetic_pendulum_histogram_experiment.py's
# BIN_EDGES_X/Y.
HISTOGRAM_BIN_EDGES_X = [-1.5, 0.0, 1.5]
HISTOGRAM_BIN_EDGES_Y = [-1.5, -0.5, 0.5, 1.5]
# Matches generate_ground_truth.jl and
# case_studies/magnetic_pendulum/main_magnetic_pendulum_pairwise_experiment.py's THRESHOLD.
PAIRWISE_THRESHOLD = 1.0


def run_magnetic_pendulum_accuracy_test(
    predictor: BaseEstimator,
    label_column: str,
    method_name: str,
    *,
    execution_target: ExecutionTarget,
) -> AccuracyComparison:
    """Run pybasin's full magnetic pendulum pipeline over the ground-truth initial conditions.

    :param predictor: Grouper to validate (``HistogramClusterer`` or
        ``PairwiseComparisonClusterer``).
    :param label_column: Which grouper's ground-truth column to compare against
        (``"label_histogram"`` or ``"label_pairwise"``).
    :param method_name: Stable method identifier used in documentation artifacts.
    :param execution_target: Device used by the sampler, solver, and grouper.
    :return: Complete comparison against Attractors.jl.
    """
    device = execution_target.value
    setup = setup_magnetic_pendulum_system(device=device)

    sampler = CsvSampler(
        GROUND_TRUTH_CSV,
        coordinate_columns=["x", "y", "vx", "vy"],
        label_column=label_column,
        device=device,
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
        system_name="magnetic_pendulum",
        method_name=method_name,
        execution_target=execution_target,
        classification_metrics=compute_classification_metrics(y_true, y_pred),
        basins=compute_basin_fraction_comparisons(y_true, y_pred),
    )


@pytest.mark.parametrize("execution_target", CPU_AND_GPU_TARGETS)
class TestMagneticPendulumHistogram:
    """`HistogramClusterer` against `GroupViaHistogram`'s labels on the magnetic pendulum."""

    def test_baseline(
        self,
        execution_target: ExecutionTarget,
        accuracy_artifact_collector: AccuracyArtifactCollector | None,
    ) -> None:
        predictor = HistogramClusterer(bin_edges=[HISTOGRAM_BIN_EDGES_X, HISTOGRAM_BIN_EDGES_Y])
        comparison = run_magnetic_pendulum_accuracy_test(
            predictor,
            "label_histogram",
            "histogram",
            execution_target=execution_target,
        )
        metrics = comparison.classification_metrics

        assert metrics.matthews_corrcoef >= 0.98, (
            f"MCC {metrics.matthews_corrcoef:.4f} < 0.98 for the histogram grouper. The three "
            "magnets converge to tight, well-separated equilibria (see generate_ground_truth.jl), "
            "and the bin edges keep a >=0.3 margin from each, so near-exact parity is expected."
        )
        assert min(metrics.f1_per_class.values()) >= 0.98
        assert all(basin.sampling_uncertainty_z_score <= 1.96 for basin in comparison.basins)
        if accuracy_artifact_collector is not None:
            accuracy_artifact_collector.add(comparison)


@pytest.mark.parametrize("execution_target", CPU_AND_GPU_TARGETS)
class TestMagneticPendulumPairwise:
    """`PairwiseComparisonClusterer` against `GroupViaPairwiseComparison`'s labels."""

    def test_baseline(
        self,
        execution_target: ExecutionTarget,
        accuracy_artifact_collector: AccuracyArtifactCollector | None,
    ) -> None:
        predictor = PairwiseComparisonClusterer(
            threshold=PAIRWISE_THRESHOLD, rescale_features=False
        )
        comparison = run_magnetic_pendulum_accuracy_test(
            predictor,
            "label_pairwise",
            "pairwise",
            execution_target=execution_target,
        )
        metrics = comparison.classification_metrics

        assert metrics.matthews_corrcoef >= 0.98, (
            f"MCC {metrics.matthews_corrcoef:.4f} < 0.98 for the pairwise grouper."
        )
        assert min(metrics.f1_per_class.values()) >= 0.98
        assert all(basin.sampling_uncertainty_z_score <= 1.96 for basin in comparison.basins)
        if accuracy_artifact_collector is not None:
            accuracy_artifact_collector.add(comparison)
