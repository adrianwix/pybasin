# pyright: basic
"""Silhouette score of each representation under the true attractor labels.

1-NN accuracy only asks whether nearest neighbours share an attractor; the silhouette asks
whether attractors form compact, well-separated groups, which is what a density clusterer
needs to recover them without labels. Attractors with fewer than ``MIN_CLASS`` samples are
excluded because no clusterer with PyBasin's minimum cluster size can recover them.

Output: ``results/silhouette_validation.csv``.
"""

import numpy as np
import pandas as pd
import representations as reps
from common import BASINS_COLLECTION_DATA_DIR, CACHE_DIR, RESULTS_DIR, load, prepare_embedding
from sklearn.metrics import silhouette_score

MIN_CLASS = 20


def main() -> None:
    rows = []
    for path in sorted(BASINS_COLLECTION_DATA_DIR.glob("bc_*.npz")):
        ds = load(path.stem)
        cdir = CACHE_DIR / "validation" / path.stem
        classes, counts = np.unique(ds.labels, return_counts=True)
        keep = np.isin(ds.labels, classes[counts >= MIN_CLASS])
        candidates = {
            "PyBasin DefaultFeatureExtractor": reps.default_features(ds),
            "Chronos-Bolt": np.load(cdir / "chronos_bolt.npy"),
            "Chronos-2": np.load(cdir / "chronos2.npy"),
            "Contrastive CNN (trained on Duffing only)": np.load(
                cdir / "contrastive_duffing_only.npy"
            ),
            "Contrastive CNN (trained on 4 case studies)": np.load(cdir / "contrastive_all.npy"),
            "Sliced-Wasserstein measure": reps.sliced_wasserstein(ds),
        }
        for name, X in candidates.items():
            Z = prepare_embedding(X[keep])
            rows.append({
                "system": path.stem,
                "representation": name,
                "silhouette": float(silhouette_score(Z, ds.labels[keep])),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "silhouette_validation.csv", index=False)
    table = df.pivot(index="representation", columns="system", values="silhouette")
    table["MEAN"] = table.mean(axis=1)
    with pd.option_context("display.width", 250):
        print(table.round(2).sort_values("MEAN", ascending=False))


if __name__ == "__main__":
    main()
