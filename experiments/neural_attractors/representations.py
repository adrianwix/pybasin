# pyright: basic
"""Trajectory representations compared in the neural attractor experiments.

Every function maps a :class:`Dataset` (trajectories ``(N, S, T)``) to an ``(N, D)`` array.
"""

import numpy as np
import torch
from common import Dataset, dataset_scale
from tstorch.settings import DYNAMICAL_SYSTEM_FEATURES_CONFIG

from pybasin.feature_extractors.default_feature_extractor import DefaultFeatureExtractor
from pybasin.feature_extractors.tstorch_feature_extractor import TorchFeatureExtractorAdapter
from pybasin.solution import Solution

N_QUANTILES = 32
N_RANDOM_DIRECTIONS = 16


def _solution(ds: Dataset) -> Solution:
    y = torch.from_numpy(ds.y).double()
    return Solution(
        initial_condition=y[:, :, 0].contiguous(),
        time=torch.from_numpy(ds.t).double(),
        y=y,
    )


def _time_steady(ds: Dataset) -> float:
    return float(ds.t[0]) - 1e-9


CASE_STUDY_FEATURES = {
    "duffing": {0: {"maximum": None, "standard_deviation": None}},
    "pendulum": {1: {"log_delta": None}},
    "friction": {1: {"absolute_maximum": None}},
    "lorenz": {0: {"mean": None}, 1: None, 2: None},
}


def case_study_features(ds: Dataset) -> np.ndarray:
    """Hand-picked per-system features from the case-study setups (expert reference)."""
    extractor = TorchFeatureExtractorAdapter(
        time_steady=_time_steady(ds),
        normalize=False,
        features=None,
        features_per_state=CASE_STUDY_FEATURES[ds.name],
    )
    return extractor.extract_features(_solution(ds)).cpu().numpy()


def default_features(ds: Dataset) -> np.ndarray:
    """PyBasin's ``DefaultFeatureExtractor``: 10 fixed statistics per state."""
    extractor = DefaultFeatureExtractor(time_steady=_time_steady(ds), normalize=False)
    return extractor.extract_features(_solution(ds)).cpu().numpy()


def dynamical_features(ds: Dataset) -> tuple[np.ndarray, list[str]]:
    """``DYNAMICAL_SYSTEM_FEATURES_CONFIG`` features, with names for ``DynamicalSystemClusterer``."""
    extractor = TorchFeatureExtractorAdapter(
        time_steady=_time_steady(ds), features=DYNAMICAL_SYSTEM_FEATURES_CONFIG, normalize=False
    )
    X = extractor.extract_features(_solution(ds)).cpu().numpy()
    return X, list(extractor.feature_names)


def sliced_wasserstein(ds: Dataset) -> np.ndarray:
    """Sliced-Wasserstein embedding of each trajectory's empirical state-space measure.

    The steady-state samples of a trajectory approximate the invariant measure of the
    attractor it lies on; that measure is independent of the trajectory's phase, so two
    trajectories on the same attractor coincide while distinct attractors (including
    symmetric images of each other) differ. Euclidean distance between quantile vectors of
    1-D projections is the sliced 2-Wasserstein distance between those measures. States are
    rescaled by pooled dataset statistics so projections weight states comparably.
    """
    loc, scale = dataset_scale(ds.y)
    z = (ds.y - loc) / scale  # (N, S, T)
    s = z.shape[1]
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((N_RANDOM_DIRECTIONS, s))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.concatenate([np.eye(s), directions])  # (P, S)
    projected = np.einsum("ps,nst->npt", directions, z)
    q = np.quantile(projected, (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES, axis=-1)
    return np.moveaxis(q, 0, -1).reshape(len(z), -1) / np.sqrt(N_QUANTILES)


def stroboscopic(ds: Dataset) -> np.ndarray:
    """Sorted states of the stroboscopic (Poincare) section at forcing phase 0.

    Only defined for periodically forced systems. A period-k orbit leaves k distinct points;
    sorting each state's section values gives a permutation-invariant set embedding.
    """
    spp = ds.samples_per_forcing_period
    if spp is None:
        raise ValueError(f"{ds.name} is autonomous; no stroboscopic section.")
    loc, scale = dataset_scale(ds.y)
    section = ((ds.y - loc) / scale)[..., ::spp]  # (N, S, K)
    return np.sort(section, axis=-1).reshape(len(section), -1)


def chronos_embeddings(
    ds: Dataset, model: str, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pooled encoder embeddings from a pretrained Chronos model, one per state.

    Chronos-Bolt encodes each state as an independent univariate series; Chronos-2 encodes
    all states jointly with cross-variate attention and returns one token sequence per state.

    Returns ``(embedding (N, S*d), loc_scale (N, 2*S))``. Chronos standardises each input
    series internally, so the embedding alone carries no offset or amplitude information;
    ``loc_scale`` holds the per-series mean and log standard deviation it removed.
    """
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model, device_map="cpu", dtype=torch.float32)
    n, s, _ = ds.y.shape
    pooled: list[torch.Tensor] = []
    with torch.inference_mode():
        if "chronos-2" in model:
            y = torch.from_numpy(ds.y)
            for i in range(0, n, batch_size):
                embs, _ = pipe.embed(y[i : i + batch_size], batch_size=batch_size)
                pooled.append(torch.stack([e.mean(dim=1).reshape(-1) for e in embs]))
        else:
            flat = torch.from_numpy(ds.y.reshape(n * s, -1))
            for i in range(0, len(flat), batch_size):
                emb, _ = pipe.embed(flat[i : i + batch_size])
                pooled.append(emb.mean(dim=1))
    emb = torch.cat(pooled).reshape(n, -1).float().numpy()
    mean = ds.y.mean(axis=-1)
    log_std = np.log(ds.y.std(axis=-1) + 1e-6)
    return emb, np.concatenate([mean, log_std], axis=1)
