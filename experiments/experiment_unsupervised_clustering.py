# pyright: basic
"""Experiment: Unsupervised clustering for basin stability across multiple systems.

Tests TorchFeatureExtractor with HDBSCAN+KNN and KMeans on:
- Pendulum system (2 attractors: FP, LC)
- Duffing oscillator (5 attractors, 1 limit cycles ¯y1, ¯y2, two period-2 limit cycles
¯y3, ¯y4 and one period-3 limit cycle ¯y5.)
- Friction system (2 attractors: FP, LC)
- Lorenz system (3 attractors: butterfly1, butterfly2, unbounded)

Uses the same methodology as experiment_torch_extractor_clustering.py:
- Comprehensive feature extraction with TorchFeatureExtractor
- Feature selection (variance threshold + correlation filtering)
- HDBSCAN clustering with KNN noise assignment
- KMeans clustering for comparison
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans  # pyright: ignore[reportAttributeAccessIssue]
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from tstorch.torch_feature_extractor import TorchFeatureExtractor

from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from case_studies.friction.setup_friction_system import setup_friction_system
from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.feature_selector.correlation_selector import CorrelationSelector
from pybasin.solution import Solution

# ==============================================================================
# Utility Functions
# ==============================================================================


def assign_noise_to_clusters(features: np.ndarray, labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Assign noise points (-1 label) to nearest clusters using KNN.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Cluster labels with -1 for noise
        k: Number of neighbors to consider

    Returns:
        Updated labels with noise assigned to clusters
    """
    labels_updated = labels.copy()
    noise_mask = labels == -1

    if not noise_mask.any():
        return labels_updated

    labeled_mask = ~noise_mask
    labeled_features = features[labeled_mask]
    labeled_labels = labels[labeled_mask]

    if len(labeled_features) == 0:
        return labels_updated

    noise_features = features[noise_mask]
    k_actual = min(k, len(labeled_features))
    nbrs = NearestNeighbors(n_neighbors=k_actual).fit(labeled_features)
    distances, indices = nbrs.kneighbors(noise_features)

    noise_indices = np.where(noise_mask)[0]
    for i, neighbor_indices in enumerate(indices):
        neighbor_labels = labeled_labels[neighbor_indices]
        unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
        labels_updated[noise_indices[i]] = most_common_label

    return labels_updated


def select_features(
    features: np.ndarray,
    feature_names: list[str],
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
) -> tuple[np.ndarray, list[str]]:
    """Apply variance threshold and correlation filtering using a sklearn Pipeline.

    Args:
        features: Feature matrix
        feature_names: List of feature names
        variance_threshold: Variance threshold
        correlation_threshold: Correlation threshold

    Returns:
        Tuple of (filtered features, remaining feature names)
    """
    # Create feature selection pipeline
    feature_selector = Pipeline([
        ("variance", VarianceThreshold(threshold=variance_threshold)),
        ("correlation", CorrelationSelector(threshold=correlation_threshold)),
    ])

    # Fit and transform features
    print("   Applying feature selection pipeline:")
    print(f"     - Variance threshold: {variance_threshold}")
    print(f"     - Correlation threshold: {correlation_threshold}")

    features_filtered = feature_selector.fit_transform(features)

    # Get masks from each step to track feature names
    var_mask = feature_selector.named_steps["variance"].get_support()
    kept_names_after_var = [
        name for name, keep in zip(feature_names, var_mask, strict=True) if keep
    ]
    print(f"   After variance filtering: {var_mask.sum()} features")

    corr_mask = feature_selector.named_steps["correlation"].get_support()
    kept_names_final = [
        name for name, keep in zip(kept_names_after_var, corr_mask, strict=True) if keep
    ]
    print(f"   After correlation filtering: {corr_mask.sum()} features")
    print(f"   Total reduction: {features.shape[1]} → {features_filtered.shape[1]}")

    return features_filtered, kept_names_final


# ==============================================================================
# Clustering Functions
# ==============================================================================


def find_optimal_hdbscan_min_cluster_size(
    features: np.ndarray,
    min_sizes: list[int] | None = None,
) -> tuple[int, dict[int, float]]:
    """Find optimal min_cluster_size using silhouette score.

    Args:
        features: Feature matrix (n_samples, n_features)
        min_sizes: List of min_cluster_size values to try. If None, will use adaptive values.

    Returns:
        Tuple of (best_min_cluster_size, scores_dict)
    """
    n_samples = len(features)

    if min_sizes is None:
        # Adaptive min_sizes based on dataset size
        min_sizes = [
            max(10, int(0.005 * n_samples)),  # 0.5%
            max(25, int(0.01 * n_samples)),  # 1%
            max(50, int(0.02 * n_samples)),  # 2%
            max(100, int(0.03 * n_samples)),  # 3%
            max(150, int(0.05 * n_samples)),  # 5%
        ]

    scores = {}
    best_score = -1
    best_min_size = min_sizes[0]

    print("   Finding optimal min_cluster_size:")
    for min_size in min_sizes:
        clusterer = HDBSCAN(
            min_cluster_size=min_size,
            min_samples=min(10, min_size // 5),
        )
        labels = clusterer.fit_predict(features)

        # Only compute silhouette if we have at least 2 clusters (excluding noise)
        unique_labels = np.unique(labels[labels != -1])
        if len(unique_labels) >= 2:
            # Compute silhouette score only on non-noise points
            mask = labels != -1
            if np.sum(mask) > 1:
                score = silhouette_score(features[mask], labels[mask])
                scores[min_size] = score
                n_clusters = len(unique_labels)
                n_noise = np.sum(labels == -1)
                print(
                    f"     min_size={min_size}: {n_clusters} clusters, {n_noise} noise, silhouette={score:.4f}"
                )

                if score > best_score:
                    best_score = score
                    best_min_size = min_size
        else:
            print(f"     min_size={min_size}: < 2 clusters, skipping")

    print(f"   Selected min_cluster_size={best_min_size} (silhouette={best_score:.4f})")
    return best_min_size, scores


def run_hdbscan_clustering(
    features: np.ndarray, min_cluster_size: int = 50, auto_tune: bool = False
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float], dict[str, float], float, int]:
    """Run HDBSCAN clustering with noise assignment.

    Args:
        features: Feature matrix
        min_cluster_size: Minimum cluster size
        auto_tune: If True, automatically find optimal min_cluster_size

    Returns:
        Tuple of (labels_raw, labels_assigned, bs_raw, bs_assigned, time, optimal_min_size)
    """
    # Auto-tune min_cluster_size if requested
    if auto_tune:
        optimal_min_size, _ = find_optimal_hdbscan_min_cluster_size(features)
        min_cluster_size = optimal_min_size

    t_start = time.perf_counter()
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min(10, min_cluster_size // 5),
    )
    labels_raw = clusterer.fit_predict(features)
    t_elapsed = time.perf_counter() - t_start

    n_samples = len(labels_raw)
    unique_labels, counts = np.unique(labels_raw, return_counts=True)

    # Basin stability with noise
    bs_raw = {}
    for label, count in zip(unique_labels, counts, strict=True):
        label_name = "Noise" if label == -1 else f"Cluster_{label}"
        bs_raw[label_name] = count / n_samples

    # Assign noise to clusters
    labels_assigned = assign_noise_to_clusters(features, labels_raw, k=5)
    unique_labels_a, counts_a = np.unique(labels_assigned, return_counts=True)

    # Basin stability with noise assigned
    bs_assigned = {}
    for label, count in zip(unique_labels_a, counts_a, strict=True):
        bs_assigned[f"Cluster_{label}"] = count / n_samples

    return labels_raw, labels_assigned, bs_raw, bs_assigned, t_elapsed, min_cluster_size


def run_kmeans_clustering(
    features: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, dict[str, float], float]:
    """Run KMeans clustering.

    Returns:
        Tuple of (labels, basin_stability, time)
    """
    t_start = time.perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    t_elapsed = time.perf_counter() - t_start

    n_samples = len(labels)
    unique_labels, counts = np.unique(labels, return_counts=True)

    bs = {}
    for label, count in zip(unique_labels, counts, strict=True):
        bs[f"Cluster_{label}"] = count / n_samples

    return labels, bs, t_elapsed


# ==============================================================================
# Main Experiment Runner
# ==============================================================================


def run_system_experiment(
    system_name: str,
    setup_fn,
    expected_n_clusters: int,
    device: Literal["cpu", "gpu"] = "cpu",
    n_jobs: int | None = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    hdbscan_min_cluster_size: int = 50,
):
    """Run clustering experiment on a single system."""
    print("\n" + "=" * 80)
    print(f"{system_name.upper()} System - Unsupervised Clustering")
    print("=" * 80)

    # Setup system
    print("\n1. Setup")
    print("-" * 40)
    setup = setup_fn()
    n = setup["n"]
    ode_system = setup["ode_system"]
    sampler = setup["sampler"]
    solver = setup["solver"]

    print(f"   System: {system_name}")
    print(f"   Samples: {n}")
    print(f"   Expected clusters: {expected_n_clusters}")
    print(f"   Feature extraction device: {device}")

    # Sample and integrate
    print("\n2. Sampling and Integration")
    print("-" * 40)
    t_start = time.perf_counter()
    initial_conditions = sampler.sample(n)
    print(f"   Initial conditions: {initial_conditions.shape}")

    time_arr, y_arr = solver.integrate(ode_system, initial_conditions)
    t_integrate = time.perf_counter() - t_start
    print(f"   Solution shape: {y_arr.shape}")
    print(f"   Integration time: {t_integrate:.3f}s")

    # Detect unbounded trajectories BEFORE feature extraction
    print("\n3. Unbounded Trajectory Detection")
    print("-" * 40)
    y_arr_np = y_arr.cpu().numpy() if hasattr(y_arr, "cpu") else np.array(y_arr)
    unbounded_mask = ~np.all(np.isfinite(y_arr_np), axis=(0, 2))
    n_unbounded = np.sum(unbounded_mask)
    n_bounded = n - n_unbounded
    print(f"   Unbounded trajectories: {n_unbounded} ({n_unbounded / n:.4f})")
    print(f"   Bounded trajectories: {n_bounded} ({n_bounded / n:.4f})")

    # Extract features only on bounded trajectories
    if n_bounded > 0:
        y_arr_bounded = y_arr[:, ~unbounded_mask, :]
        ics_bounded = initial_conditions[~unbounded_mask]
        solution = Solution(initial_condition=ics_bounded, time=time_arr, y=y_arr_bounded)
    else:
        solution = None

    # Feature extraction (only bounded trajectories)
    print("\n4. Feature Extraction with TorchFeatureExtractor (bounded only)")
    print("-" * 40)

    if n_bounded > 0 and solution is not None:
        print(f"   Processing {n_bounded} bounded trajectories")
        feature_extractor = TorchFeatureExtractor(
            time_steady=950.0,
            features="minimal",
            normalize=True,
            device=device,
            n_jobs=n_jobs,
            impute_method="extreme",
        )
        print("   Feature config: minimal (efficient subset of tsfresh features)")
        print(f"   Device: {device}, n_jobs: {n_jobs or 'all'}")

        t_extract_start = time.perf_counter()
        features = feature_extractor.extract_features(solution)
        t_extract = time.perf_counter() - t_extract_start

        features_np = features.detach().cpu().numpy()
        print(f"   Features shape: {features.shape}")
        print(f"   Extraction time: {t_extract:.3f}s")
        print(f"   Total features extracted: {len(feature_extractor.feature_names)}")
    else:
        print("   No bounded trajectories to extract features from")
        features_np = None
        t_extract = 0.0
        feature_extractor = None

    # Add JAX dynamical features BEFORE filtering (only bounded)
    print("\n5. Adding JAX Dynamical Features (bounded only)")
    print("-" * 40)

    if n_bounded > 0 and features_np is not None and feature_extractor is not None:
        all_features = features_np
        all_feature_names = feature_extractor.feature_names
    else:
        all_features = None
        all_feature_names = []

    # Feature selection (only bounded)
    print("\n6. Feature Selection (on tsfresh+JAX features, bounded only)")
    print("-" * 40)

    if n_bounded > 0 and all_features is not None:
        t_selection_start = time.perf_counter()
        features_final, kept_names_final = select_features(
            all_features,
            all_feature_names,
            variance_threshold,
            correlation_threshold,
        )
        t_selection = time.perf_counter() - t_selection_start
        print(f"   Reduction: {all_features.shape[1]} → {features_final.shape[1]}")
        print(f"   Selection time: {t_selection:.3f}s")
    else:
        features_final = None
        t_selection = 0.0

    # HDBSCAN clustering on bounded trajectories only
    print("\n7. HDBSCAN Clustering (on bounded trajectories only)")
    print("-" * 40)
    if n_bounded > 0 and features_final is not None:
        (
            labels_h_raw_bounded,
            labels_h_assigned_bounded,
            bs_h_raw_bounded,
            bs_h_assigned_bounded,
            t_hdbscan,
            optimal_min_size,
        ) = run_hdbscan_clustering(
            features_final, min_cluster_size=hdbscan_min_cluster_size, auto_tune=True
        )
        print(f"   Clustering time: {t_hdbscan:.4f}s")
        print(f"   Optimal min_cluster_size: {optimal_min_size}")
        print(f"   Clusters found: {len([k for k in bs_h_raw_bounded if k != 'Noise'])}")

        if "Noise" in bs_h_raw_bounded:
            print(f"   Noise: {bs_h_raw_bounded['Noise']:.4f}")

        print("\n   HDBSCAN Basin Stability (bounded only, noise assigned):")
        for label, frac in sorted(bs_h_assigned_bounded.items()):
            count = int(frac * n_bounded)
            print(f"     {label}: {frac:.4f} ({count} samples)")

        # Combine unbounded + HDBSCAN labels for full dataset
        labels_h_combined = np.full(n, -999, dtype=int)
        labels_h_combined[unbounded_mask] = -1  # Label unbounded as -1
        labels_h_combined[~unbounded_mask] = labels_h_assigned_bounded

        # Calculate basin stability for combined labels
        unique_labels, counts = np.unique(labels_h_combined, return_counts=True)
        bs_h_combined = {}
        for label, count in zip(unique_labels, counts, strict=True):
            if label == -1:
                bs_h_combined["unbounded"] = count / n
            else:
                bs_h_combined[f"Cluster_{label}"] = count / n

        print("\n   Combined Basin Stability (unbounded + HDBSCAN):")
        for label, frac in sorted(bs_h_combined.items()):
            count = int(frac * n)
            print(f"     {label}: {frac:.4f} ({count} samples)")

        bs_h_assigned = bs_h_combined
    else:
        print("   No bounded trajectories")
        bs_h_assigned = {"unbounded": n_unbounded / n} if n_unbounded > 0 else {}
        t_hdbscan = 0.0

    # KMeans clustering on bounded trajectories
    # Adjust cluster count if there are unbounded trajectories
    kmeans_n_clusters = expected_n_clusters - 1 if n_unbounded > 0 else expected_n_clusters
    print(f"\n8. KMeans Clustering (n_clusters={kmeans_n_clusters}, bounded only)")
    print("-" * 40)
    if n_bounded > 0 and features_final is not None:
        labels_k_bounded, bs_k_bounded, t_kmeans = run_kmeans_clustering(
            features_final, kmeans_n_clusters
        )
        print(f"   Clustering time: {t_kmeans:.4f}s")

        print("\n   KMeans Basin Stability (bounded only):")
        for label, frac in sorted(bs_k_bounded.items()):
            count = int(frac * n_bounded)
            print(f"     {label}: {frac:.4f} ({count} samples)")

        # Combine with unbounded for full dataset
        labels_k_combined = np.full(n, -1, dtype=int)
        labels_k_combined[~unbounded_mask] = labels_k_bounded

        unique_labels_k, counts_k = np.unique(labels_k_combined, return_counts=True)
        bs_k = {}
        for label, count in zip(unique_labels_k, counts_k, strict=True):
            if label == -1:
                bs_k["unbounded"] = count / n
            else:
                bs_k[f"Cluster_{label}"] = count / n

        print("\n   Combined KMeans Basin Stability (unbounded + KMeans):")
        for label, frac in sorted(bs_k.items()):
            count = int(frac * n)
            print(f"     {label}: {frac:.4f} ({count} samples)")
    else:
        bs_k = {"unbounded": 1.0}
        t_kmeans = 0.0

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total samples: {n}")
    print(f"Unbounded: {n_unbounded}, Bounded: {n_bounded}")
    if feature_extractor:
        print(f"Total features extracted: {len(feature_extractor.feature_names)}")
    if features_final is not None:
        print(f"Features after selection + JAX: {features_final.shape[1]}")
    print(f"Integration time: {t_integrate:.3f}s")
    print(f"Feature extraction time: {t_extract:.3f}s")
    print(f"Feature selection time: {t_selection:.3f}s")
    print(f"HDBSCAN time: {t_hdbscan:.4f}s")
    print(f"KMeans time: {t_kmeans:.4f}s")

    return {
        "system_name": system_name,
        "n_samples": n,
        "n_unbounded": n_unbounded,
        "n_bounded": n_bounded,
        "n_features_original": len(feature_extractor.feature_names) if feature_extractor else 0,
        "n_jax_features": 0,
        "n_features_selected": features_final.shape[1] if features_final is not None else 0,
        "hdbscan_bs": bs_h_assigned,
        "kmeans_bs": bs_k,
        "times": {
            "integration": t_integrate,
            "extraction": t_extract,
            "selection": t_selection,
            "hdbscan": t_hdbscan,
            "kmeans": t_kmeans,
        },
    }


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unsupervised clustering experiment for basin stability"
    )
    parser.add_argument(
        "--system",
        choices=["pendulum", "duffing", "friction", "lorenz", "all"],
        default="all",
        help="System to run (default: all)",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device")
    parser.add_argument("--n-jobs", type=int, default=None, help="CPU workers")
    parser.add_argument("--variance-threshold", type=float, default=0.01)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50)

    args = parser.parse_args()

    # Expected basin stability from integration tests (supervised KNN with known attractors)
    expected_bs = {
        "pendulum": {"FP": 0.152, "LC": 0.848},
        "duffing": {"y1": 0.1908, "y2": 0.4994, "y3": 0.0278, "y4": 0.022, "y5": 0.26},
        "friction": {"FP": 0.304, "LC": 0.696},
        "lorenz": {"butterfly1": 0.0894, "butterfly2": 0.08745, "unbounded": 0.82315},
    }

    systems = {
        "pendulum": (setup_pendulum_system, 2),
        "duffing": (setup_duffing_oscillator_system, 5),
        "friction": (setup_friction_system, 2),
        "lorenz": (setup_lorenz_system, 3),
    }

    if args.system == "all":
        systems_to_run = systems.items()
    else:
        systems_to_run = [(args.system, systems[args.system])]

    results = []
    for system_name, (setup_fn, n_clusters) in systems_to_run:
        result = run_system_experiment(
            system_name=system_name,
            setup_fn=setup_fn,
            expected_n_clusters=n_clusters,
            device=args.device,
            n_jobs=args.n_jobs,
            variance_threshold=args.variance_threshold,
            correlation_threshold=args.correlation_threshold,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        )
        results.append(result)

    # Print final comparison
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON ACROSS ALL SYSTEMS")
    print("=" * 80)

    for result in results:
        system_name = result["system_name"]
        print(f"\n{system_name.upper()}:")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Features: {result['n_features_original']} → {result['n_features_selected']}")

        print("\n  Expected (Supervised KNN from integration tests):")
        for label, frac in sorted(expected_bs[system_name].items()):
            count = int(frac * result["n_samples"])
            print(f"    {label}: {frac:.4f} ({count} samples)")

        if result["hdbscan_bs"]:
            print("\n  HDBSCAN (noise assigned):")
            for label, frac in sorted(result["hdbscan_bs"].items()):
                count = int(frac * result["n_samples"])
                print(f"    {label}: {frac:.4f} ({count} samples)")

        print("\n  KMeans:")
        for label, frac in sorted(result["kmeans_bs"].items()):
            count = int(frac * result["n_samples"])
            print(f"    {label}: {frac:.4f} ({count} samples)")


if __name__ == "__main__":
    main()
