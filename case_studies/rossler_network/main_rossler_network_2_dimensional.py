# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Two-dimensional parameter study for Rössler network basin stability.

This study investigates basin stability as a function of:
1. Coupling constant K (within stability interval)
2. Network rewiring probability p (Watts-Strogatz topology)

Implements Section 2.2.3 from the Menck et al. supplementary material.

Expected behavior:
- Basin stability increases with rewiring probability p
- Regular lattice (p=0): S_B ~ 0.30
- Small-world regime (p=0.2-0.5): S_B ~ 0.49-0.55
- Random network (p=1.0): S_B ~ 0.60

Work in progress: The code crashes when using WSL probably due to memory issues
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import networkx as nx
import numpy as np
import torch
from jax import Array

from case_studies.rossler_network.rossler_network_jax_ode import (
    RosslerNetworkJaxODE,
    RosslerNetworkParams,
)
from case_studies.rossler_network.synchronization_classifier import (
    SynchronizationClassifier,
)
from case_studies.rossler_network.synchronization_feature_extractor import (
    SynchronizationFeatureExtractor,
)
from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.study_params import CustomStudyParams, ParamAssignment, RunConfig
from pybasin.types import StudyResult
from pybasin.utils import generate_filename, time_execution

logger = logging.getLogger(__name__)


def build_edge_arrays_from_networkx(graph: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build edge index arrays from NetworkX graph for sparse Laplacian.

    :param graph: NetworkX graph.
    :return: Tuple of (source node indices, target node indices), each shape (2*E,).
    """
    edges_i: list[int] = []
    edges_j: list[int] = []
    for i, j in graph.edges():
        edges_i.append(int(i))
        edges_j.append(int(j))
        edges_i.append(int(j))
        edges_j.append(int(i))

    return np.array(edges_i, dtype=np.int32), np.array(edges_j, dtype=np.int32)


def compute_stability_interval(graph: Any) -> tuple[float, float]:
    """Compute stability interval for coupling constant K.

    :param graph: NetworkX graph.
    :return: Tuple of (k_min, k_max) bounds of the stability interval.
    """
    ALPHA_1 = 0.1232
    ALPHA_2 = 4.663

    laplacian = nx.laplacian_matrix(graph).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(eigenvalues)

    lambda_min = eigenvalues[1]
    lambda_max = eigenvalues[-1]

    k_min = ALPHA_1 / lambda_min
    k_max = ALPHA_2 / lambda_max

    return float(k_min), float(k_max)


def rossler_stop_event(t: Array, y: Array, args: Any, **kwargs: Any) -> Any:
    """Event function to stop integration when amplitude exceeds threshold."""
    MAX_VAL = 400
    max_abs_y = jnp.max(jnp.abs(y))
    return MAX_VAL - max_abs_y


def main() -> None:
    """Run the two-dimensional parameter study.

    Uses ``CustomStudyParams`` to define all (p, K) combinations upfront as a
    single flat study. For each rewiring probability *p*, a Watts-Strogatz
    network is generated and *K* values are sampled from the stability interval.
    Each ``RunConfig`` assigns a fully-configured ``ode_system`` (with the
    correct edges and coupling constant), so the ``BasinStabilityStudy`` handles
    everything in one ``run()`` call.
    """
    P_VALUES = np.arange(0.0, 1.05, 0.05)
    N_NODES = 400
    K_DEGREE = 8
    N_EDGES = N_NODES * K_DEGREE // 2
    N_SAMPLES = 500
    N_K_VALUES = 11
    SEED = 42
    ROSSLER_A = 0.2
    ROSSLER_B = 0.2
    ROSSLER_C = 7.0

    save_dir = Path("results_2d")
    save_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"\n{'=' * 80}")
    logger.info("Two-Dimensional Parameter Study: K vs p")
    logger.info(f"{'=' * 80}")
    logger.info(f"Network: N={N_NODES}, k={K_DEGREE}")
    logger.info(f"Samples per (K,p) pair: {N_SAMPLES}")
    logger.info(f"Number of K values per p: {N_K_VALUES}")
    logger.info(f"Number of p values: {len(P_VALUES)}")
    logger.info(f"Total parameter combinations: {len(P_VALUES) * N_K_VALUES}")
    logger.info(f"{'=' * 80}\n")

    # ------------------------------------------------------------------
    # Build all (p, K) parameter combinations as CustomStudyParams
    # ------------------------------------------------------------------
    configs: list[RunConfig] = []

    for p in P_VALUES:
        p_float = float(p)
        graph = nx.watts_strogatz_graph(n=N_NODES, k=K_DEGREE, p=p_float, seed=SEED)
        edges_i, edges_j = build_edge_arrays_from_networkx(graph)
        k_min, k_max = compute_stability_interval(graph)
        k_values = np.linspace(k_min, k_max, N_K_VALUES)

        logger.info(
            f"p={p_float:.2f}: K ∈ [{k_min:.4f}, {k_max:.4f}], edges={graph.number_of_edges()}"
        )

        for K in k_values:
            ode_params: RosslerNetworkParams = {
                "a": ROSSLER_A,
                "b": ROSSLER_B,
                "c": ROSSLER_C,
                "K": float(K),
                "edges_i": jnp.array([edges_i]),
                "edges_j": jnp.array(edges_j),
                "N": N_NODES,
            }
            ode = RosslerNetworkJaxODE(ode_params)
            configs.append(
                RunConfig(
                    assignments=[ParamAssignment("ode_system", ode)],
                    study_label={"p": p_float, "K": float(K)},
                )
            )

    study_params = CustomStudyParams(configs)

    # ------------------------------------------------------------------
    # Shared components (unchanged across runs)
    # ------------------------------------------------------------------
    # The first config's ODE serves as the base; it gets replaced per run.
    base_ode: RosslerNetworkJaxODE = configs[0].assignments[0].value

    min_limits = [-15.0] * N_NODES + [-15.0] * N_NODES + [-4.0] * N_NODES
    max_limits = [15.0] * N_NODES + [15.0] * N_NODES + [35.0] * N_NODES

    sampler = UniformRandomSampler(
        min_limits=min_limits,
        max_limits=max_limits,
        device=device,
    )

    solver = JaxSolver(
        t_span=(0, 1000),
        t_steps=1000,
        device=device,
        rtol=1e-3,
        atol=1e-6,
        cache_dir=".pybasin_cache/rossler_network",
        event_fn=rossler_stop_event,
    )

    feature_extractor = SynchronizationFeatureExtractor(
        n_nodes=N_NODES,
        time_steady=950,
        device=device,
    )

    sync_classifier = SynchronizationClassifier(epsilon=1.5)

    # ------------------------------------------------------------------
    # Run the study
    # ------------------------------------------------------------------
    study = BasinStabilityStudy(
        n=N_SAMPLES,
        ode_system=base_ode,
        sampler=sampler,
        solver=solver,
        feature_extractor=feature_extractor,
        estimator=sync_classifier,
        study_params=study_params,
        output_dir=str(save_dir),
        verbose=False,
    )

    results = study.run()

    # ------------------------------------------------------------------
    # Group results by p for summary and JSON export
    # ------------------------------------------------------------------
    p_groups: dict[float, list[StudyResult]] = defaultdict(list)
    for result in results:
        p_groups[result["study_label"]["p"]].append(result)

    p_summaries: list[dict[str, Any]] = []
    for p_val in sorted(p_groups.keys()):
        group = p_groups[p_val]
        k_values_in_group: list[float] = [r["study_label"]["K"] for r in group]
        sync_values: list[float] = [r["basin_stability"].get("synchronized", 0.0) for r in group]
        p_summaries.append(
            {
                "p": p_val,
                "seed": SEED,
                "n_nodes": N_NODES,
                "k_degree": K_DEGREE,
                "n_edges": N_EDGES,
                "n_samples": N_SAMPLES,
                "stability_interval": {
                    "K_min": min(k_values_in_group),
                    "K_max": max(k_values_in_group),
                },
                "K_values": k_values_in_group,
                "basin_stabilities": [r["basin_stability"] for r in group],
                "mean_sb": float(np.mean(sync_values)),
                "study_labels": [r["study_label"] for r in group],
            }
        )

    filename = generate_filename(name="2d_parameter_study", file_extension="json")
    filepath = save_dir / filename
    with open(filepath, "w") as f:
        json.dump(p_summaries, f, indent=2)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY: Basin Stability vs Rewiring Probability")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'p':>6} | {'Mean S_B':>9} | {'K_min':>8} | {'K_max':>8} | {'Edges':>6}")
    logger.info("-" * 55)

    mean_sbs: list[float] = []
    for result in p_summaries:
        p_val = result["p"]
        mean_sb = result["mean_sb"]
        k_min = result["stability_interval"]["K_min"]
        k_max = result["stability_interval"]["K_max"]
        n_edges = result["n_edges"]
        logger.info(
            f"{p_val:>6.2f} | {mean_sb:>9.3f} | {k_min:>8.4f} | {k_max:>8.4f} | {n_edges:>6}"
        )
        mean_sbs.append(mean_sb)

    logger.info("-" * 55)
    logger.info(f"\nResults saved to: {filepath}")

    print("\n" + "=" * 80)
    print("Key Finding:")
    print("=" * 80)
    if mean_sbs[-1] > mean_sbs[0]:
        print("✓ Basin stability INCREASES with rewiring probability")
        print(f"  Regular lattice (p={P_VALUES[0]:.1f}): S_B = {mean_sbs[0]:.3f}")
        print(f"  Random network (p={P_VALUES[-1]:.1f}):  S_B = {mean_sbs[-1]:.3f}")
        if mean_sbs[0] > 0:
            print(f"  Relative increase: {(mean_sbs[-1] / mean_sbs[0] - 1) * 100:.1f}%")
        else:
            print(f"  Absolute increase: {mean_sbs[-1] - mean_sbs[0]:.3f}")
    else:
        print("⚠ Unexpected: Basin stability did not increase with p")
    print("=" * 80)


if __name__ == "__main__":
    time_execution("main_rossler_network_2_dimensional.py", main)
