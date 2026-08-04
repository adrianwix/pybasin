# pyright: basic
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
"""Experiment: tsfresh feature filtering for attractor clustering.

This experiment analyzes which tsfresh features are most effective for clustering
time series trajectories using DBSCAN to identify attractors for basin stability
estimation.

The experiment:
1. Runs each case study with uniform sampling across the state space
2. Uses KNN with templates to get ground truth attractor labels
3. Extracts comprehensive tsfresh features
4. Uses tsfresh's feature selection to identify the most relevant features
5. Evaluates clustering performance with different feature subsets using DBSCAN
6. Reports the best features for each case study

Goal: Determine which features are best for unsupervised attractor discovery.
"""

import logging
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute

sys.path.insert(0, str(Path(__file__).parent.parent))

from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from case_studies.friction.setup_friction_system import setup_friction_system
from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system

from pybasin.solution import Solution
from pybasin.types import SetupProperties

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*did not have any finite values.*")
warnings.filterwarnings("ignore", message=".*Filling with zeros.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.getLogger("tsfresh").setLevel(logging.ERROR)


def get_case_studies() -> list[tuple[str, SetupProperties]]:
    """Get all case study configurations from their setup functions."""
    return [
        ("Pendulum", setup_pendulum_system()),
        ("Duffing", setup_duffing_oscillator_system()),
        ("Lorenz", setup_lorenz_system()),
        ("Friction", setup_friction_system()),
    ]


def solve_trajectories(
    setup: SetupProperties,
    initial_conditions: torch.Tensor,
) -> Solution:
    """Solve ODE system for given initial conditions."""
    solver = setup.get("solver")
    assert solver is not None, "Setup must include a solver for ground truth labeling"

    ode_system = setup["ode_system"]

    time_arr, y_arr = solver.integrate(ode_system, initial_conditions)

    params_dict: dict[str, float] = dict(ode_system.params)

    return Solution(
        initial_condition=initial_conditions,
        time=time_arr,
        y=y_arr,
        model_params=params_dict,
    )


def get_ground_truth_labels(
    setup: SetupProperties,
    solution: Solution,
) -> np.ndarray:
    """Get ground truth labels using the case study's KNN classifier with templates.

    This uses the exact same classification pipeline as the actual case studies.
    """
    solver = setup.get("solver")
    ode_system = setup["ode_system"]
    feature_extractor = setup.get("feature_extractor")
    template_integrator = setup.get("template_integrator")

    assert solver is not None, "Setup must include a solver for ground truth labeling"
    assert feature_extractor is not None, (
        "Setup must include a feature_extractor for ground truth labeling"
    )
    assert template_integrator is not None, (
        "Setup must include a template_integrator for ground truth labeling"
    )

    template_y0 = template_integrator.template_y0
    template_labels = template_integrator.labels

    template_tensor = torch.tensor(template_y0, dtype=torch.float32, device=solver.device)
    time_arr, y_arr = solver.integrate(ode_system, template_tensor)
    template_solution = Solution(
        initial_condition=template_tensor,
        time=time_arr,
        y=y_arr,
        model_params=dict(ode_system.params),
    )

    feature_extractor.reset_scaler()  # type: ignore[attr-defined]

    template_features = feature_extractor.extract_features(template_solution)
    template_features_np = template_features.cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(template_features_np, template_labels)

    sample_features = feature_extractor.extract_features(solution)
    sample_features_np = sample_features.cpu().numpy()

    labels = knn.predict(sample_features_np)

    return labels


def extract_tsfresh_features(
    solution: Solution,
    time_steady: float,
    fc_parameters: Any = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract tsfresh features from solution trajectories.

    Returns:
        Tuple of (features_df, valid_mask) where valid_mask indicates which
        samples were valid (no inf/nan values).
    """
    y = solution.y
    time_np = solution.time.cpu().numpy()

    if time_steady > 0:
        idx_steady = int(np.searchsorted(time_np, time_steady, side="right"))
        y = y[idx_steady:]
        time_np = time_np[idx_steady:]

    n_timesteps, n_batch, n_states = y.shape
    y_np = y.cpu().numpy()

    valid_mask = np.all(np.isfinite(y_np), axis=(0, 2))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        raise ValueError("All trajectories contain inf/nan values")

    if len(valid_indices) < n_batch:
        print(f"    Filtering out {n_batch - len(valid_indices)} divergent trajectories")

    data_list: list[dict[str, Any]] = []
    for new_idx, orig_idx in enumerate(valid_indices):
        for time_idx in range(n_timesteps):
            row: dict[str, Any] = {
                "id": new_idx,
                "time": time_idx,
            }
            for state_idx in range(n_states):
                row[f"state_{state_idx}"] = y_np[time_idx, orig_idx, state_idx]
            data_list.append(row)

    df_pivot = pd.DataFrame(data_list)

    if fc_parameters is None:
        fc_parameters = EfficientFCParameters()

    features_df: pd.DataFrame = cast(
        pd.DataFrame,
        extract_features(
            df_pivot,
            column_id="id",
            column_sort="time",
            default_fc_parameters=fc_parameters,
            n_jobs=1,
            disable_progressbar=True,
        ),
    )

    impute(features_df)

    return features_df, valid_mask


def evaluate_clustering(
    features: np.ndarray,
    ground_truth: np.ndarray,
    eps_values: list[float] | None = None,
    min_samples: int = 3,
) -> dict[str, Any]:
    """Evaluate DBSCAN clustering with different epsilon values."""
    if eps_values is None:
        eps_values = [0.3, 0.5, 1.0, 2.0, 3.0]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    best_result: dict[str, Any] = {
        "ari": -1.0,
        "nmi": -1.0,
        "silhouette": -1.0,
        "n_clusters": 0,
        "eps": 0.0,
    }

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        predicted = dbscan.fit_predict(features_scaled)

        n_clusters = len(set(predicted)) - (1 if -1 in predicted else 0)
        if n_clusters < 2:
            continue

        ari = adjusted_rand_score(ground_truth, predicted)
        nmi = normalized_mutual_info_score(ground_truth, predicted)

        non_noise_mask = predicted != -1
        if non_noise_mask.sum() > n_clusters:
            sil = silhouette_score(features_scaled[non_noise_mask], predicted[non_noise_mask])
        else:
            sil = -1.0

        if ari > best_result["ari"]:
            best_result = {
                "ari": ari,
                "nmi": nmi,
                "silhouette": sil,
                "n_clusters": n_clusters,
                "eps": eps,
                "predicted": predicted,
            }

    return best_result


def rank_features_by_variance(features_df: pd.DataFrame, top_k: int = 20) -> list[str]:
    """Rank features by their variance (simple unsupervised approach)."""
    variances = features_df.var().sort_values(ascending=False)
    return list(variances.head(top_k).index)


def rank_features_by_target_relevance(
    features_df: pd.DataFrame,
    target: pd.Series,
    top_k: int = 20,
) -> list[str]:
    """Use tsfresh's feature selection to rank features by relevance to target.

    This uses statistical tests to find features that are significantly
    correlated with the target (attractor labels).
    """
    try:
        selected_features = select_features(features_df, target, n_jobs=1)
        return list(selected_features.columns[:top_k])
    except Exception as e:
        print(f"    Warning: tsfresh feature selection failed: {e}")
        return rank_features_by_variance(features_df, top_k)


def analyze_feature_importance_unsupervised(
    features_df: pd.DataFrame,
    ground_truth: np.ndarray,
) -> pd.DataFrame:
    """Analyze which individual features work best for DBSCAN clustering."""
    results = []

    for col in features_df.columns:
        feature_values = features_df[[col]].values

        if np.isnan(feature_values).any() or np.isinf(feature_values).any():
            continue

        scaler = StandardScaler()
        try:
            scaled = scaler.fit_transform(feature_values)
        except Exception:
            continue

        if np.std(scaled) < 1e-10:
            continue

        best_ari = -1.0
        best_eps = 0.0
        for eps in [0.1, 0.3, 0.5, 1.0, 2.0]:
            dbscan = DBSCAN(eps=eps, min_samples=2)
            predicted = dbscan.fit_predict(scaled)

            n_clusters = len(set(predicted)) - (1 if -1 in predicted else 0)
            if n_clusters >= 2:
                ari = adjusted_rand_score(ground_truth, predicted)
                if ari > best_ari:
                    best_ari = ari
                    best_eps = eps

        results.append({"feature": col, "best_ari": best_ari, "best_eps": best_eps})

    return pd.DataFrame(results).sort_values("best_ari", ascending=False)


def run_experiment_for_case_study(
    name: str,
    setup: SetupProperties,
    n_samples: int = 500,
) -> dict[str, Any]:
    """Run the full experiment for a single case study."""
    template_integrator = setup.get("template_integrator")
    assert template_integrator is not None, "Setup must include a template_integrator"
    template_labels = template_integrator.labels
    feature_extractor = setup.get("feature_extractor")
    assert feature_extractor is not None, "Setup must include a feature_extractor"

    print(f"\n{'=' * 80}")
    print(f"Case Study: {name}")
    print(f"{'=' * 80}")
    print(f"Templates: {len(template_integrator.template_y0)} attractors")
    print(f"Labels: {template_labels}")

    print(f"\n1. Sampling {n_samples} initial conditions from state space...")
    sampler = setup["sampler"]
    initial_conditions = sampler.sample(n_samples)
    print(f"   Shape: {initial_conditions.shape}")

    print("\n2. Solving ODE trajectories...")
    t_start = time.perf_counter()
    solution = solve_trajectories(setup, initial_conditions)
    t_solve = time.perf_counter() - t_start
    print(f"   Solved in {t_solve:.2f}s")

    print("\n3. Getting ground truth labels using KNN with templates...")
    ground_truth = get_ground_truth_labels(setup, solution)
    unique_labels, counts = np.unique(ground_truth, return_counts=True)
    print("   Label distribution:")
    for label, count in zip(unique_labels, counts, strict=True):
        print(f"     {label}: {count} ({100 * count / len(ground_truth):.1f}%)")

    time_steady = feature_extractor.time_steady

    print("\n4. Extracting tsfresh features (EfficientFCParameters)...")
    t_start = time.perf_counter()
    features_df, valid_mask = extract_tsfresh_features(
        solution,
        time_steady,
        fc_parameters=EfficientFCParameters(),
    )
    t_extract = time.perf_counter() - t_start
    print(f"   Extracted {features_df.shape[1]} features in {t_extract:.2f}s")

    ground_truth = ground_truth[valid_mask]

    print("\n--- Feature Analysis ---")

    print("\n5. Individual Feature Performance (unsupervised DBSCAN):")
    feature_importance = analyze_feature_importance_unsupervised(features_df, ground_truth)
    top_features = feature_importance.head(15)
    print(top_features.to_string(index=False))

    print("\n6. Clustering with All Features:")
    all_features_result = evaluate_clustering(features_df.values, ground_truth)
    print(f"   ARI: {all_features_result['ari']:.4f}")
    print(f"   NMI: {all_features_result['nmi']:.4f}")
    print(f"   Clusters found: {all_features_result['n_clusters']}")
    print(f"   Best eps: {all_features_result['eps']}")

    print("\n7. Clustering with Top-K Features by Variance:")
    for k in [5, 10, 20]:
        top_k_features = rank_features_by_variance(features_df, top_k=k)
        if len(top_k_features) < k:
            continue
        result = evaluate_clustering(features_df[top_k_features].values, ground_truth)
        print(
            f"   Top-{k}: ARI={result['ari']:.4f}, NMI={result['nmi']:.4f}, "
            f"clusters={result['n_clusters']}"
        )

    print("\n8. Clustering with tsfresh Selected Features:")
    label_to_int = {label: i for i, label in enumerate(template_labels)}
    ground_truth_int = np.array([label_to_int[lbl] for lbl in ground_truth])
    target = pd.Series(ground_truth_int, index=features_df.index)
    selected_features = rank_features_by_target_relevance(features_df, target, top_k=20)
    if len(selected_features) >= 2:
        result = evaluate_clustering(features_df[selected_features].values, ground_truth)
        print(f"   Selected {len(selected_features)} features")
        print(
            f"   ARI: {result['ari']:.4f}, NMI: {result['nmi']:.4f}, clusters={result['n_clusters']}"
        )
        print("   Top selected features:")
        for feat in selected_features[:10]:
            print(f"     - {feat}")

    print("\n9. Clustering with Minimal Feature Sets:")
    minimal_feature_sets = [
        ["state_0__mean", "state_0__standard_deviation"],
        ["state_0__maximum", "state_0__minimum"],
        ["state_0__variance", "state_1__variance"],
        ["state_0__mean", "state_0__variance", "state_1__mean", "state_1__variance"],
    ]

    for feature_set in minimal_feature_sets:
        available_features = [f for f in feature_set if f in features_df.columns]
        if len(available_features) >= 2:
            result = evaluate_clustering(features_df[available_features].values, ground_truth)
            print(f"   {available_features}: ARI={result['ari']:.4f}")

    return {
        "name": name,
        "n_templates": len(template_integrator.template_y0),
        "n_samples": len(ground_truth),
        "n_features": features_df.shape[1],
        "top_individual_features": feature_importance.head(10).to_dict("records"),
        "all_features_ari": all_features_result["ari"],
        "selected_features": selected_features[:10] if selected_features else [],
    }


def run_full_experiment() -> None:
    """Run the experiment across all case studies."""
    print("=" * 80)
    print("TSFRESH FEATURE FILTERING EXPERIMENT")
    print("Goal: Find best features for DBSCAN clustering of ODE attractors")
    print("=" * 80)

    case_studies = get_case_studies()
    all_results = []

    for name, setup in case_studies:
        try:
            result = run_experiment_for_case_study(name, setup, n_samples=500)
            all_results.append(result)
        except Exception as e:
            print(f"\nError in {name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY: Best Features Across All Case Studies")
    print("=" * 80)

    individual_feature_counts: dict[str, int] = {}
    selected_feature_counts: dict[str, int] = {}

    for result in all_results:
        print(f"\n{result['name']}:")
        print("  Best individual features:")
        seen_individual: set[str] = set()
        for feat_info in result["top_individual_features"][:5]:
            feat_name = feat_info["feature"]
            ari = feat_info["best_ari"]
            eps = feat_info["best_eps"]
            print(f"    - {feat_name}: ARI={ari:.4f}, eps={eps}")

            base_name = feat_name.split("__")[1] if "__" in feat_name else feat_name
            if base_name not in seen_individual:
                seen_individual.add(base_name)
                individual_feature_counts[base_name] = (
                    individual_feature_counts.get(base_name, 0) + 1
                )

        print("  Top selected features (tsfresh):")
        seen_selected: set[str] = set()
        for feat_name in result["selected_features"][:5]:
            print(f"    - {feat_name}")
            base_name = feat_name.split("__")[1] if "__" in feat_name else feat_name
            if base_name not in seen_selected:
                seen_selected.add(base_name)
                selected_feature_counts[base_name] = selected_feature_counts.get(base_name, 0) + 1

    print("\n" + "-" * 40)
    print("Most Frequently Useful Individual Features (across all systems):")
    sorted_individual = sorted(individual_feature_counts.items(), key=lambda x: -x[1])
    for feat, count in sorted_individual[:15]:
        print(f"  {feat}: appears in {count} case studies")

    print("\n" + "-" * 40)
    print("Most Frequently Selected Features by tsfresh (across all systems):")
    sorted_selected = sorted(selected_feature_counts.items(), key=lambda x: -x[1])
    for feat, count in sorted_selected[:15]:
        print(f"  {feat}: appears in {count} case studies")


if __name__ == "__main__":
    run_full_experiment()
