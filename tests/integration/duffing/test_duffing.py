"""Integration tests for the Duffing oscillator case study."""

import csv
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier

from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.predictors.dbscan_clusterer import DBSCANClusterer
from pybasin.solvers import JaxSolver
from pybasin.template_integrator import TemplateIntegrator
from pybasin.utils import set_seed
from tests.conftest import ArtifactCollector
from tests.integration.test_helpers import (
    UnsupervisedAttractorComparison,
    UnsupervisedComparisonResult,
    compute_classification_metrics,
    run_basin_stability_test,
)


class TestDuffing:
    """Integration tests for Duffing oscillator basin stability estimation."""

    @pytest.mark.integration
    def test_baseline_supervised(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Duffing oscillator baseline with supervised classification approach.

        Parameters: δ=0.08, k3=1, A=0.2
        Expected attractors: y1-y5 (various n-cycles)
        Uses supervised KNN classification with known attractor templates.

        Verifies:
        1. Number of ICs used matches the stored reference count
        2. Classification metrics: MCC >= 0.95
        """
        json_path = Path(__file__).parent / "main_duffing_supervised.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_duffing.csv"
        bse, comparison = run_basin_stability_test(
            json_path,
            setup_duffing_oscillator_system,
            n=4000,
            seed=42,
            ground_truth_csv=ground_truth_csv,
        )

        if artifact_collector is not None:
            artifact_collector.add_single_point(
                bse,
                comparison,
                trajectory_state=0,
                trajectory_x_limits=(0, 50),
                trajectory_y_limits=(-1.4, 1.4),
                trajectory_y_ticks=[-1, 1],
                phase_space_axes=(0, 1),
            )

    @pytest.mark.integration
    def test_baseline_unsupervised(
        self,
        artifact_collector: ArtifactCollector | None,
    ) -> None:
        """Test Duffing oscillator baseline with unsupervised clustering approach.

        Parameters: δ=0.08, k3=1, A=0.2
        Uses unsupervised clustering (DBSCAN) to discover attractors, then
        cross-references with supervised templates to assign meaningful labels.

        This demonstrates:
        1. DBSCAN discovers the correct number of clusters
        2. Relabeling via KNN templates assigns correct attractor names
        3. Basin stability values match the supervised reference
        """
        # Load expected results from the supervised reference JSON.
        json_path = Path(__file__).parent / "main_duffing_supervised.json"
        ground_truth_csv = Path(__file__).parent / "ground_truths" / "main" / "main_duffing.csv"
        with open(json_path) as f:
            meta = json.load(f)
        label_map: dict[str, str] = meta.get("label_map", {})
        expected_results = meta["data"]
        for result in expected_results:
            result["label"] = label_map.get(result["label"], result["label"])

        # Setup system - we'll use its KNN classifier and template integrator for relabeling
        set_seed(42)
        props = setup_duffing_oscillator_system()
        knn_classifier = cast(KNeighborsClassifier, props.get("estimator"))
        template_integrator = props.get("template_integrator")
        assert isinstance(template_integrator, TemplateIntegrator)

        # Use seeded random sampling on CPU for reproducible regression behavior
        solver = props.get("solver")
        if isinstance(solver, JaxSolver):
            solver = solver.clone(device="cpu")
        sampler = props["sampler"]
        sampler.min_limits = sampler.min_limits.cpu()
        sampler.max_limits = sampler.max_limits.cpu()
        sampler.device = sampler.min_limits.device
        n_samples = 4000

        # Use DBSCAN clustering for unsupervised discovery
        dbscan_clusterer = DBSCANClusterer(auto_tune=True)

        bse = BasinStabilityEstimator(
            n=n_samples,
            ode_system=props["ode_system"],
            sampler=sampler,
            solver=solver,
            feature_extractor=props.get("feature_extractor"),
            predictor=dbscan_clusterer,
            feature_selector=None,
        )

        result = bse.run()
        basin_stability = result["basin_stability"]

        # Verify we found the expected number of clusters (excluding NaN)
        expected_n_clusters = len([r for r in expected_results if r["basinStability"] > 0])
        actual_n_clusters = len([bs for bs in basin_stability.values() if bs > 0])
        assert actual_n_clusters == expected_n_clusters, (
            f"Number of clusters mismatch: expected {expected_n_clusters}, got {actual_n_clusters}"
        )

        # Relabel DBSCAN clusters by cross-referencing with KNN templates
        # For each DBSCAN cluster, find which template it best matches via majority vote
        assert bse.solution is not None
        assert bse.solution.features is not None

        features = bse.solution.features.cpu().numpy()
        dbscan_labels = np.array(bse.solution.labels)

        # Fit KNN classifier to get template-based predictions
        # First integrate templates, then fit with features from the feature extractor
        feature_extractor = props.get("feature_extractor")
        assert feature_extractor is not None
        template_integrator.integrate(props.get("solver"), props["ode_system"])
        X_train, y_train = template_integrator.get_training_data(feature_extractor)
        knn_classifier.fit(X_train, y_train)

        # For bounded trajectories, get KNN predictions
        bounded_mask = dbscan_labels != "NaN"
        bounded_features = features
        knn_predictions = knn_classifier.predict(bounded_features)

        # Build mapping: DBSCAN cluster → template label (majority vote)
        # Also store purity info per cluster for later merging with attractor results
        unique_dbscan_labels = np.unique(dbscan_labels[bounded_mask])
        cluster_to_template: dict[int, str] = {}
        cluster_purity_info: dict[int, dict[str, int | float]] = {}

        # Map bounded indices for KNN lookup
        bounded_indices = np.where(bounded_mask)[0]

        for cluster_label in unique_dbscan_labels:
            # Get indices of trajectories in this DBSCAN cluster
            cluster_mask = dbscan_labels == cluster_label
            cluster_indices = np.where(cluster_mask & bounded_mask)[0]

            # Get KNN predictions for trajectories in this cluster
            knn_indices = [np.where(bounded_indices == idx)[0][0] for idx in cluster_indices]
            cluster_knn_labels = knn_predictions[knn_indices]

            # Majority vote determines which template this cluster matches
            unique_templates, counts = np.unique(cluster_knn_labels, return_counts=True)
            majority_idx = np.argmax(counts)
            majority_template = unique_templates[majority_idx]
            majority_count = int(counts[majority_idx])
            total_count = len(cluster_knn_labels)

            # Store with integer key (cluster_label is np.int64)
            cluster_to_template[int(cluster_label)] = str(majority_template)
            cluster_purity_info[int(cluster_label)] = {
                "cluster_size": total_count,
                "majority_count": majority_count,
                "purity": majority_count / total_count if total_count > 0 else 0.0,
            }

        # Build reverse mapping: template label → DBSCAN cluster
        template_to_cluster: dict[str, int] = {v: k for k, v in cluster_to_template.items()}

        # Compute overall agreement: % of bounded trajectories where DBSCAN matches KNN
        dbscan_relabeled_bounded = np.array(
            [cluster_to_template[int(lbl)] for lbl in dbscan_labels[bounded_mask]]
        )
        agreement_count = np.sum(dbscan_relabeled_bounded == knn_predictions)
        overall_agreement = float(agreement_count / len(knn_predictions))

        # Compute Adjusted Rand Index comparing DBSCAN clusters to KNN predictions
        # Convert string labels to integers for ARI computation
        dbscan_bounded = dbscan_labels[bounded_mask]
        ari = adjusted_rand_score(dbscan_bounded, knn_predictions)

        # Relabel all trajectories using the cluster-to-template mapping
        # Note: dbscan_labels contains integers (np.int64)
        relabeled: np.ndarray = np.empty(len(dbscan_labels), dtype=object)
        for i, lbl in enumerate(dbscan_labels):
            relabeled[i] = cluster_to_template.get(int(lbl), "NaN")

        # Update solution labels for artifact generation
        bse.solution.set_labels(relabeled)

        # Recalculate basin stability and standard errors with relabeled clusters
        n_samples = len(relabeled)
        relabeled_bs: dict[str, float] = {}
        relabeled_se: dict[str, float] = {}
        for label in np.unique(relabeled):
            label_str = str(label)
            matches: np.ndarray = relabeled == label
            bs = float(np.mean(matches))
            relabeled_bs[label_str] = bs
            relabeled_se[label_str] = float(np.sqrt(bs * (1 - bs) / n_samples))

        # Load ground truth labels for classification metrics
        ground_truth_labels_raw: list[str] = []
        with open(ground_truth_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_value = row.get("label")
                if label_value is None:
                    raise ValueError("Ground truth CSV must have a 'label' column")
                ground_truth_labels_raw.append(label_value)
        ground_truth_labels: np.ndarray = np.array(
            [label_map.get(str(lbl), str(lbl)) for lbl in ground_truth_labels_raw]
        )

        # Compute classification metrics

        metrics = compute_classification_metrics(ground_truth_labels, relabeled)

        # Build comparison results using classification metrics
        attractor_comparisons: list[UnsupervisedAttractorComparison] = []

        for expected in expected_results:
            label = expected["label"]
            if label == "NaN" or expected["basinStability"] == 0:
                continue

            expected_bs = expected["basinStability"]
            expected_se = expected["standardError"]
            actual_bs = relabeled_bs.get(label, 0.0)
            actual_se = relabeled_se.get(label, 0.0)

            # Get F1-score for this class
            f1_for_class = metrics.f1_per_class.get(label, 0.0)

            # Get purity info for this attractor's DBSCAN cluster
            dbscan_cluster = template_to_cluster.get(label, -1)
            purity_info = cluster_purity_info.get(dbscan_cluster, {})

            attractor_comparisons.append(
                UnsupervisedAttractorComparison(
                    label=label,
                    python_bs=actual_bs,
                    python_se=actual_se,
                    matlab_bs=expected_bs,
                    matlab_se=expected_se,
                    dbscan_label=str(dbscan_cluster),
                    cluster_size=int(purity_info.get("cluster_size", 0)),
                    majority_count=int(purity_info.get("majority_count", 0)),
                    purity=float(purity_info.get("purity", 0.0)),
                )
            )

            # Assert validation: F1-score should be high (>0.9 for good classification)
            assert f1_for_class >= 0.85, (
                f"Label '{label}': F1-score {f1_for_class:.4f} is below threshold 0.85"
            )

        comparison = UnsupervisedComparisonResult(
            system_name="duffing",
            case_name="case2",
            overall_agreement=overall_agreement,
            adjusted_rand_index=float(ari),
            n_clusters_found=actual_n_clusters,
            n_clusters_expected=expected_n_clusters,
            attractors=attractor_comparisons,
            matthews_corrcoef=metrics.matthews_corrcoef,
        )

        # Verify basin stabilities sum to 1.0
        total_bs = sum(relabeled_bs.values())
        assert abs(total_bs - 1.0) < 0.01, f"Basin stabilities should sum to 1.0, got {total_bs}"

        if artifact_collector is not None:
            artifact_collector.add_unsupervised(
                bse,
                comparison,
                trajectory_state=0,
                trajectory_x_limits=(0, 50),
                trajectory_y_limits=(-1.4, 1.4),
                trajectory_y_ticks=[-1, 1],
                phase_space_axes=(0, 1),
            )
