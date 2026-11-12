# pyright: basic
"""Label-free attractor tracking across the Duffing forcing amplitude ``A``.

At each ``A`` the same initial conditions are integrated, attractors are discovered with the
PyBasin default clusterer on a chosen representation, and every cluster is summarised by
its median embedding (its "signature"). Clusters at neighbouring ``A`` values are linked by
minimum-cost matching of signatures; a link is accepted only when the signature distance is
below ``LINK_FACTOR`` times the larger within-cluster spread of the two clusters, so a new or
vanished attractor starts or ends a branch. Each cluster's period is read off the stroboscopic
section of its medoid trajectory (number of distinct section points).

Output: ``results/continuation_<representation>.csv`` and ``results/continuation_<representation>.png``.
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import representations as reps
import torch
from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE
from common import RESULTS_DIR, Dataset, prepare_embedding, pybasin_default_clusterer
from generate_data import DUFFING_STEPS, DUFFING_WINDOW, SAMPLES_PER_FORCING_PERIOD
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pybasin.solvers import JaxSolver

AMPLITUDES = np.round(np.linspace(0.12, 0.30, 10), 3)
N_IC = 1500
LINK_FACTOR = 3.0
PERIOD_TOL = 1e-3


def integrate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y0 = torch.tensor(rng.uniform([-1, -0.5], [1, 1], size=(N_IC, 2)), dtype=torch.float32)
    params = torch.tensor([[0.08, 1.0, a] for a in AMPLITUDES], dtype=torch.float32)
    solver = JaxSolver(
        t_span=(0.0, 1000.0),
        t_steps=DUFFING_STEPS,
        t_eval=DUFFING_WINDOW,
        device="cpu",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,
    )
    t, y = solver.integrate(DuffingJaxODE({"delta": 0.08, "k3": 1, "A": 0.2}), y0, params)
    y = y.cpu().numpy()[..., :-1].reshape(N_IC, len(AMPLITUDES), 2, -1).transpose(1, 0, 2, 3)
    return t.cpu().numpy()[:-1], y, y0.numpy()  # y: (P, N, S, T)


def period(traj: np.ndarray) -> int:
    section = traj[:, ::SAMPLES_PER_FORCING_PERIOD].T  # (K, S)
    distinct: list[np.ndarray] = []
    for p in section:
        if all(np.linalg.norm(p - q) > PERIOD_TOL * (1 + np.linalg.norm(q)) for q in distinct):
            distinct.append(p)
    return len(distinct) if len(distinct) < len(section) else -1


def embed_all(t: np.ndarray, y: np.ndarray, representation: str) -> np.ndarray:
    pooled = Dataset("duffing", y.reshape(-1, *y.shape[2:]), t, np.array([]), 2 * math.pi)
    X = {"sliced_wasserstein": reps.sliced_wasserstein, "stroboscopic": reps.stroboscopic}[
        representation
    ](pooled)
    return X.reshape(y.shape[0], y.shape[1], -1)


def main(representation: str) -> None:
    t, y, _ = integrate()
    E = embed_all(t, y, representation)
    clusters: list[dict] = []
    for p, a in enumerate(AMPLITUDES):
        labels = np.asarray(
            pybasin_default_clusterer().fit_predict(torch.from_numpy(prepare_embedding(E[p])))
        )
        for k in np.unique(labels):
            members = np.flatnonzero(labels == k)
            sig = np.median(E[p, members], axis=0)
            d = np.linalg.norm(E[p, members] - sig, axis=1)
            medoid = members[np.argmin(d)]
            clusters.append({
                "A": a,
                "p": p,
                "cluster": int(k),
                "bs": len(members) / N_IC,
                "spread": float(np.median(d)),
                "sig": sig,
                "period": period(y[p, medoid]),
                "x_max": float(y[p, medoid, 0].max()),
            })
    df = pd.DataFrame(clusters)

    branch_of: dict[int, int] = {}
    n_branches = 0
    for p in range(len(AMPLITUDES)):
        cur = df.index[df.p == p].to_numpy()
        prev = df.index[df.p == p - 1].to_numpy() if p > 0 else np.array([], dtype=int)
        linked: set[int] = set()
        if len(prev):
            D = cdist(np.stack(df.loc[prev, "sig"]), np.stack(df.loc[cur, "sig"]))
            for i, j in zip(*linear_sum_assignment(D), strict=True):
                tol = LINK_FACTOR * max(df.at[prev[i], "spread"], df.at[cur[j], "spread"], 1e-3)
                if D[i, j] <= tol:
                    branch_of[cur[j]] = branch_of[prev[i]]
                    linked.add(cur[j])
        for c in cur:
            if c not in linked:
                branch_of[c] = n_branches
                n_branches += 1
    df["branch"] = [branch_of[i] for i in df.index]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = df.drop(columns=["sig"])
    out.to_csv(RESULTS_DIR / f"continuation_{representation}.csv", index=False)
    print(out.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for b, g in df.groupby("branch"):
        g = g.sort_values("A")
        periods = sorted(set(g.period))
        tag = "/".join("chaotic" if q == -1 else f"P{q}" for q in periods)
        ax.plot(g.A, g.bs, "o-", label=f"branch {b} ({tag})")
    ax.set_xlabel("forcing amplitude A")
    ax.set_ylabel("basin stability")
    ax.set_title(f"Duffing: label-free attractor tracking ({representation})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"continuation_{representation}.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        default="sliced_wasserstein",
        choices=["sliced_wasserstein", "stroboscopic"],
    )
    main(parser.parse_args().representation)
