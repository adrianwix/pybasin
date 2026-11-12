# pyright: basic
"""Zero-shot validation on BasinsCollection systems that no trained encoder has seen.

Contrastive encoders are trained only on PyBasin case-study data: one on Duffing alone and
one on all four case studies. Chronos models are pretrained and used as is. Every
BasinsCollection system is then embedded and scored with the same fixed clusterer as in
``evaluate.py``; labels come from Attractors.jl's recurrence mapper (``basins_collection/
generate.jl``).

Usage::

    python evaluate_validation.py            # every system present in basins_collection/data

Output: ``results/scores_validation.csv``.
"""

import contrastive
import numpy as np
import pandas as pd
import representations as reps
from common import (
    BASINS_COLLECTION_DATA_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    SYSTEMS,
    Condition,
    load,
    prepare_embedding,
    score,
    scores_from_clusters,
)
from evaluate import CHRONOS_2, CHRONOS_BOLT, cached, contrastive_model

from pybasin.predictors import DynamicalSystemClusterer


def main() -> None:
    names = sorted(p.stem for p in BASINS_COLLECTION_DATA_DIR.glob("bc_*.npz"))
    case_studies = {name: load(name) for name in SYSTEMS}
    encoders = {
        "duffing_only": (
            "Contrastive CNN (trained on Duffing only)",
            contrastive_model("duffing_only", [case_studies["duffing"]], Condition()),
        ),
        "all": (
            "Contrastive CNN (trained on 4 case studies)",
            contrastive_model("all", list(case_studies.values()), Condition()),
        ),
    }
    rows: list[dict] = []
    for name in names:
        ds = load(name)
        print(f"{name}: {ds.y.shape}, {len(np.unique(ds.labels))} attractors")
        cdir = CACHE_DIR / "validation" / name
        bolt = cached(
            cdir / "chronos_bolt.npy", lambda ds=ds: reps.chronos_embeddings(ds, CHRONOS_BOLT)[0]
        )
        c2 = cached(cdir / "chronos2.npy", lambda ds=ds: reps.chronos_embeddings(ds, CHRONOS_2)[0])
        loc_scale = np.concatenate([ds.y.mean(axis=-1), np.log(ds.y.std(axis=-1) + 1e-6)], axis=1)
        candidates: dict[str, list[np.ndarray]] = {
            "PyBasin DefaultFeatureExtractor": [reps.default_features(ds)],
            "Chronos-Bolt": [bolt],
            "Chronos-Bolt + loc/scale": [bolt, loc_scale],
            "Chronos-2": [c2],
            "Chronos-2 + loc/scale": [c2, loc_scale],
            **{
                label: [
                    cached(
                        cdir / f"contrastive_{key}.npy",
                        lambda ds=ds, m=model: contrastive.embed(m, ds),
                    )
                ]
                for key, (label, model) in encoders.items()
            },
            "Sliced-Wasserstein measure": [reps.sliced_wasserstein(ds)],
        }
        if ds.samples_per_forcing_period is not None:
            candidates["Stroboscopic section"] = [reps.stroboscopic(ds)]
        for rep_name, blocks in candidates.items():
            rows.append({"system": name, "representation": rep_name, **score(blocks, ds.labels)})

        X_dyn, feature_names = reps.dynamical_features(ds)
        dsc = DynamicalSystemClusterer()
        dsc.set_feature_names([
            n.replace("__min_lag_2__peak_threshold_0.3", "") for n in feature_names
        ])
        clusters = np.asarray(dsc.fit_predict(X_dyn))
        rows.append({
            "system": name,
            "representation": "PyBasin DynamicalSystemClusterer",
            **scores_from_clusters(
                prepare_embedding(X_dyn), ds.labels, clusters, dims=X_dyn.shape[1]
            ),
        })
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "scores_validation.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
