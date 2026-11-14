# pyright: basic
"""How much of the gap between representations is the clustering step?

Phase-locked periodic attractors collapse to near-identical points in any good embedding;
HDBSCAN then splits such a cluster at numerical-noise scale. This script re-scores every
cached embedding of the 11 BasinsCollection systems after merging clusters closer than a few
fixed epsilons (``common.merge_close_clusters``), the same value for every system.

Run ``evaluate_validation.py`` and ``evaluate_extended.py`` first (their caches are reused).
Output: ``results/clusterer_sensitivity.csv``.
"""

import numpy as np
import pandas as pd
import representations as reps
import torch
from common import (
    BASINS_COLLECTION_DATA_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    load,
    merge_close_clusters,
    prepare_embedding,
    pybasin_default_clusterer,
)
from sklearn.metrics import adjusted_rand_score

EPSILONS = (0.0, 0.1, 0.5, 1.0, 2.0)
CACHED = {
    "Chronos-Bolt": ("validation", "chronos_bolt"),
    "Chronos-2": ("validation", "chronos2"),
    "Contrastive CNN (duffing_only)": ("extended/w1_n0", "cnn_duffing_only"),
    "Contrastive CNN (all)": ("extended/w1_n0", "cnn_all"),
    "Set encoder (duffing_only)": ("extended/w1_n0", "set_duffing_only"),
    "Set encoder (all)": ("extended/w1_n0", "set_all"),
    "Set encoder (untrained)": ("extended/w1_n0", "set_untrained"),
}


def main() -> None:
    rows = []
    for path in sorted(BASINS_COLLECTION_DATA_DIR.glob("bc_*.npz")):
        ds = load(path.stem)
        print(path.stem)
        embeddings = {
            name: np.load(CACHE_DIR / sub / path.stem / f"{file}.npy")
            for name, (sub, file) in CACHED.items()
        }
        embeddings["PyBasin DefaultFeatureExtractor"] = reps.default_features(ds)
        embeddings["Sliced-Wasserstein measure"] = reps.sliced_wasserstein(ds)
        for name, X in embeddings.items():
            Z = prepare_embedding(X)
            base = np.asarray(pybasin_default_clusterer().fit_predict(torch.from_numpy(Z)))
            for eps in EPSILONS:
                clusters = merge_close_clusters(Z, base, eps) if eps else base
                rows.append({
                    "system": path.stem,
                    "representation": name,
                    "epsilon": eps,
                    "ari": float(adjusted_rand_score(ds.labels, clusters)),
                    "n_found": len(np.unique(clusters)),
                    "n_true": len(np.unique(ds.labels)),
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "clusterer_sensitivity.csv", index=False)
    table = df.pivot_table(index="representation", columns="epsilon", values="ari", aggfunc="mean")
    with pd.option_context("display.width", 200):
        print(table.round(2).sort_values(EPSILONS[0], ascending=False))


if __name__ == "__main__":
    main()
