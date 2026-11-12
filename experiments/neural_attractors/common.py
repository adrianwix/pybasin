# pyright: basic
"""Shared data loading, preprocessing and scoring for the neural attractor experiments."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from pybasin.predictors import HDBSCANClusterer, MinClusterSizeSearch, TunedClusterer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BASINS_COLLECTION_DATA_DIR = ROOT / "basins_collection" / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
SYSTEMS = ("duffing", "pendulum", "friction", "lorenz")
MAX_EMBED_DIMS = 16


@dataclass(frozen=True)
class Dataset:
    name: str
    y: np.ndarray  # (N, S, T) float32
    t: np.ndarray  # (T,)
    labels: np.ndarray  # (N,) str
    forcing_period: float

    @property
    def samples_per_forcing_period(self) -> int | None:
        if np.isnan(self.forcing_period):
            return None
        dt = float(self.t[1] - self.t[0])
        return round(self.forcing_period / dt)


@dataclass(frozen=True)
class Condition:
    """Observation condition applied to the saved steady-state window.

    ``window_fraction`` keeps the trailing fraction of the window; ``noise`` adds Gaussian
    noise with standard deviation ``noise`` times each state's pooled standard deviation.
    """

    window_fraction: float = 1.0
    noise: float = 0.0

    @property
    def key(self) -> str:
        return f"w{self.window_fraction:g}_n{self.noise:g}"


def load(name: str, condition: Condition | None = None) -> Dataset:
    condition = condition or Condition()
    data_dir = BASINS_COLLECTION_DATA_DIR if name.startswith("bc_") else DATA_DIR
    d = np.load(data_dir / f"{name}.npz")
    y, t = d["y"], d["t"]
    forcing_period = float(np.asarray(d["forcing_period"]).reshape(-1)[0])
    if condition.window_fraction < 1.0:
        n_keep = int(round(y.shape[-1] * condition.window_fraction))
        if not np.isnan(forcing_period):
            spp = round(forcing_period / float(t[1] - t[0]))
            n_keep = max(spp, (n_keep // spp) * spp)
        y, t = y[..., -n_keep:], t[-n_keep:]
    if condition.noise > 0:
        rng = np.random.default_rng(0)
        scale = y.std(axis=(0, 2), keepdims=True)
        y = y + condition.noise * scale * rng.standard_normal(y.shape).astype(np.float32)
    labels = np.asarray(d["labels"]).reshape(-1).astype(str)
    return Dataset(name, np.ascontiguousarray(y, dtype=np.float32), t, labels, forcing_period)


def dataset_scale(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust per-state location/scale pooled over all trajectories and time.

    Pooled (dataset-level) scaling keeps between-trajectory differences in offset, sign and
    amplitude intact, which per-series instance normalisation would erase.
    """
    flat = np.moveaxis(y, 1, 0).reshape(y.shape[1], -1)
    loc = np.median(flat, axis=1)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    scale = np.maximum(q75 - q25, 1e-8)
    return loc[None, :, None], scale[None, :, None]


def prepare_embedding(X: np.ndarray) -> np.ndarray:
    """Standardise columns, drop constant ones, and PCA-reduce to ``MAX_EMBED_DIMS``."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    std = X.std(axis=0)
    X = X[:, std > 1e-12]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    if X.shape[1] > MAX_EMBED_DIMS:
        X = PCA(n_components=MAX_EMBED_DIMS, random_state=0).fit_transform(X)
    return X


def knn_balanced_accuracy(X: np.ndarray, labels: np.ndarray) -> float:
    """5-fold 1-NN balanced accuracy: how much attractor identity the embedding holds."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = np.empty_like(labels)
    for train, test in skf.split(X, labels):
        clf = KNeighborsClassifier(n_neighbors=1).fit(X[train], labels[train])
        pred[test] = clf.predict(X[test])
    return float(balanced_accuracy_score(labels, pred))


def pybasin_default_clusterer() -> TunedClusterer:
    return TunedClusterer(
        estimator=HDBSCANClusterer(assign_noise=True, backend="cpu"),
        search=MinClusterSizeSearch(),
    )


def basin_stability_error(labels: np.ndarray, clusters: np.ndarray) -> float:
    """Largest absolute basin-stability error over true attractors after optimal matching.

    Clusters are matched one-to-one to attractors by maximum overlap; an attractor left
    without a cluster is estimated as 0, so merging two attractors shows up as an error.
    """
    classes = np.unique(labels)
    cluster_ids = np.unique(clusters)
    overlap = np.array([
        [np.sum((labels == c) & (clusters == k)) for k in cluster_ids] for c in classes
    ])
    rows, cols = linear_sum_assignment(-overlap)
    n = len(labels)
    est = dict.fromkeys(classes, 0.0)
    for r, c in zip(rows, cols, strict=True):
        est[classes[r]] = float(np.sum(clusters == cluster_ids[c])) / n
    return max(abs(float(np.mean(labels == c)) - est[c]) for c in classes)


def score(blocks: list[np.ndarray], labels: np.ndarray) -> dict[str, float]:
    """Score a representation given as feature blocks.

    Each block is standardised and PCA-reduced on its own before concatenation, so a
    low-dimensional block (e.g. per-series location/scale) keeps equal standing next to a
    high-dimensional neural embedding.
    """
    Z = np.concatenate([prepare_embedding(b) for b in blocks], axis=1)
    clusters = np.asarray(pybasin_default_clusterer().fit_predict(torch.from_numpy(Z)))
    return scores_from_clusters(Z, labels, clusters, dims=sum(b.shape[1] for b in blocks))


def scores_from_clusters(
    Z: np.ndarray, labels: np.ndarray, clusters: np.ndarray, dims: int
) -> dict[str, float]:
    return {
        "dims": dims,
        "knn_bacc": knn_balanced_accuracy(Z, labels),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "n_found": len(np.unique(clusters)),
        "n_true": len(np.unique(labels)),
        "max_bs_err": basin_stability_error(labels, clusters),
    }
