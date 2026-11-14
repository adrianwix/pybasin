# pyright: basic
"""Set encoder against the other representations on all 15 systems, under one condition.

Encoders (CNN and set encoder, each trained on Duffing only or on the four case studies)
are always trained on the clean full-window case-study data; degraded conditions measure how
the learned representation transfers to shorter or noisier observations. Rows where the
system was part of the encoder's training data are flagged ``in_training``; BasinsCollection
systems are never in training data. Chronos is evaluated in ``evaluate.py`` and
``evaluate_validation.py`` for the baseline condition only.

Usage::

    python evaluate_extended.py                  # baseline condition
    python evaluate_extended.py --window 0.25    # trailing quarter of every window
    python evaluate_extended.py --noise 0.05     # noise at 5 % of pooled state std

Output: ``results/scores_extended_<condition>.csv``.
"""

import argparse

import contrastive
import numpy as np
import pandas as pd
import representations as reps
import set_encoder
import torch
from common import (
    BASINS_COLLECTION_DATA_DIR,
    CACHE_DIR,
    RESULTS_DIR,
    SYSTEMS,
    Condition,
    Dataset,
    load,
    prepare_embedding,
    score,
    scores_from_clusters,
)
from evaluate import cached, contrastive_model

from pybasin.predictors import DynamicalSystemClusterer

TRAINING_SETS = {"duffing_only": ("duffing",), "all": SYSTEMS}


def set_model(key: str, case_studies: dict[str, Dataset]) -> set_encoder.SetEncoder:
    path = CACHE_DIR / Condition().key / "models" / f"set_{key}.pt"
    if path.exists():
        model = set_encoder.SetEncoder()
        model.load_state_dict(torch.load(path))
        return model.eval()
    names = TRAINING_SETS[key]
    print(f"  training set encoder '{key}' on {list(names)}")
    model = set_encoder.train([case_studies[n] for n in names])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def evaluate(condition: Condition) -> pd.DataFrame:
    clean = {name: load(name) for name in SYSTEMS}
    cnn = {
        k: contrastive_model(k, [clean[n] for n in TRAINING_SETS[k]], Condition())
        for k in TRAINING_SETS
    }
    sets = {k: set_model(k, clean) for k in TRAINING_SETS}
    sets["untrained"] = set_encoder.untrained()

    names = list(SYSTEMS) + sorted(p.stem for p in BASINS_COLLECTION_DATA_DIR.glob("bc_*.npz"))
    rows: list[dict] = []
    for name in names:
        ds = load(name, condition)
        print(f"[{condition.key}] {name}: {ds.y.shape}")
        cdir = CACHE_DIR / "extended" / condition.key / name
        neural: dict[str, tuple[np.ndarray, bool]] = {}
        for key, model in cnn.items():
            X = cached(cdir / f"cnn_{key}.npy", lambda ds=ds, m=model: contrastive.embed(m, ds))
            neural[f"Contrastive CNN ({key})"] = (X, name in TRAINING_SETS[key])
        for key, model in sets.items():
            X = cached(cdir / f"set_{key}.npy", lambda ds=ds, m=model: set_encoder.embed(m, ds))
            neural[f"Set encoder ({key})"] = (X, name in TRAINING_SETS.get(key, ()))

        candidates: dict[str, tuple[np.ndarray, bool]] = {
            "PyBasin DefaultFeatureExtractor": (reps.default_features(ds), False),
            "Sliced-Wasserstein measure": (reps.sliced_wasserstein(ds), False),
            **neural,
        }
        if ds.samples_per_forcing_period is not None:
            candidates["Stroboscopic section"] = (reps.stroboscopic(ds), False)
        for rep_name, (X, in_training) in candidates.items():
            rows.append({
                "system": name,
                "representation": rep_name,
                "in_training": in_training,
                **score([X], ds.labels),
            })

        X_dyn, feature_names = reps.dynamical_features(ds)
        dsc = DynamicalSystemClusterer()
        dsc.set_feature_names([
            n.replace("__min_lag_2__peak_threshold_0.3", "") for n in feature_names
        ])
        clusters = np.asarray(dsc.fit_predict(X_dyn))
        rows.append({
            "system": name,
            "representation": "PyBasin DynamicalSystemClusterer",
            "in_training": False,
            **scores_from_clusters(
                prepare_embedding(X_dyn), ds.labels, clusters, dims=X_dyn.shape[1]
            ),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    args = parser.parse_args()
    condition = Condition(window_fraction=args.window, noise=args.noise)
    df = evaluate(condition)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"scores_extended_{condition.key}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
