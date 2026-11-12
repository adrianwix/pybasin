# pyright: basic
"""Score every representation on every system with one fixed unsupervised clustering setup.

Usage::

    python evaluate.py                    # baseline condition
    python evaluate.py --window 0.25      # trailing 25 % of the steady-state window
    python evaluate.py --noise 0.05       # additive noise, 5 % of pooled state std

Embeddings are cached under ``cache/<condition>/<system>/`` so re-scoring is cheap.
Results are written to ``results/scores_<condition>.csv``.
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import contrastive
import numpy as np
import pandas as pd
import representations as reps
import torch
from common import (
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

from pybasin.predictors import DynamicalSystemClusterer

CHRONOS_BOLT = "amazon/chronos-bolt-small"
CHRONOS_2 = "amazon/chronos-2"


def cached(path: Path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    X = compute()
    print(
        f"    computed {path.parent.name}/{path.stem} {X.shape} in {time.perf_counter() - start:.1f}s"
    )
    np.save(path, X)
    return X


def contrastive_model(
    name: str, train_sets: list[Dataset], condition: Condition
) -> contrastive.DilatedEncoder:
    """Encoders are always trained on the clean full-window data; degraded conditions
    measure how the learned representation transfers to shorter or noisier observations."""
    path = CACHE_DIR / Condition().key / "models" / f"{name}.pt"
    if not path.exists() and condition != Condition():
        raise RuntimeError("Run the baseline condition first to train the encoders.")
    if path.exists():
        model = contrastive.DilatedEncoder()
        model.load_state_dict(torch.load(path))
        return model.eval()
    print(f"  training contrastive encoder '{name}' on {[d.name for d in train_sets]}")
    model = contrastive.train(train_sets)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def evaluate(condition: Condition) -> pd.DataFrame:
    datasets = {name: load(name, condition) for name in SYSTEMS}
    rows: list[dict] = []
    for name, ds in datasets.items():
        print(f"[{condition.key}] {name}: {ds.y.shape}")
        cdir = CACHE_DIR / condition.key / name

        def get(rep: str, compute: Callable[[], np.ndarray], _cdir: Path = cdir) -> np.ndarray:
            return cached(_cdir / f"{rep}.npy", compute)

        bolt = get("chronos_bolt", lambda ds=ds: reps.chronos_embeddings(ds, CHRONOS_BOLT)[0])
        c2 = get("chronos2", lambda ds=ds: reps.chronos_embeddings(ds, CHRONOS_2)[0])
        loc_scale = get(
            "loc_scale",
            lambda ds=ds: np.concatenate(
                [ds.y.mean(axis=-1), np.log(ds.y.std(axis=-1) + 1e-6)], axis=1
            ),
        )
        others = [d for n, d in datasets.items() if n != name]
        cl_all = get(
            "contrastive_all",
            lambda ds=ds: contrastive.embed(
                contrastive_model("all", list(datasets.values()), condition), ds
            ),
        )
        cl_loso = get(
            "contrastive_loso",
            lambda ds=ds, others=others, name=name: contrastive.embed(
                contrastive_model(f"without_{name}", others, condition), ds
            ),
        )

        candidates: dict[str, list[np.ndarray]] = {
            "case-study features (expert)": [
                get("case_study", lambda ds=ds: reps.case_study_features(ds))
            ],
            "PyBasin DefaultFeatureExtractor": [
                get("default", lambda ds=ds: reps.default_features(ds))
            ],
            "Chronos-Bolt": [bolt],
            "Chronos-Bolt + loc/scale": [bolt, loc_scale],
            "Chronos-2": [c2],
            "Chronos-2 + loc/scale": [c2, loc_scale],
            "Contrastive CNN (trained on all 4)": [cl_all],
            "Contrastive CNN (system held out)": [cl_loso],
            "Sliced-Wasserstein measure": [
                get("sliced_wasserstein", lambda ds=ds: reps.sliced_wasserstein(ds))
            ],
        }
        if ds.samples_per_forcing_period is not None:
            candidates["Stroboscopic section"] = [
                get("stroboscopic", lambda ds=ds: reps.stroboscopic(ds))
            ]

        for rep_name, blocks in candidates.items():
            rows.append({"system": name, "representation": rep_name, **score(blocks, ds.labels)})

        X_dyn, names = reps.dynamical_features(ds)
        dsc = DynamicalSystemClusterer()
        # The adapter embeds default calculator parameters in the column names, which
        # DynamicalSystemClusterer's base-name lookup does not recognise.
        dsc.set_feature_names([n.replace("__min_lag_2__peak_threshold_0.3", "") for n in names])
        clusters = np.asarray(dsc.fit_predict(X_dyn))

        rows.append({
            "system": name,
            "representation": "PyBasin DynamicalSystemClusterer",
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
    out = RESULTS_DIR / f"scores_{condition.key}.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.round(3).to_string(index=False))
    print(f"wrote {out}")
