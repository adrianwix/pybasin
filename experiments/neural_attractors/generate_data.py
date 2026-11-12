# pyright: basic
"""Integrate the labelled ground-truth initial conditions of four case studies and save
fixed-length steady-state windows for representation-learning experiments.

Every system uses the ODE, parameters, integration horizon and tolerances of its case study
setup; only the save region differs (a dense window at the end of the horizon so that neural
encoders see well-resolved oscillations). Labels are the MATLAB bSTAB reference labels from
``tests/integration/*/ground_truths``.

Duffing's window starts at a multiple of the forcing period ``2*pi`` and holds exactly
``SAMPLES_PER_FORCING_PERIOD`` samples per period, so ``y[..., ::SAMPLES_PER_FORCING_PERIOD]``
is the stroboscopic (Poincare) section at forcing phase 0.

Output: ``experiments/neural_attractors/data/<system>.npz`` with arrays ``y`` (N, S, T),
``t`` (T,), ``y0`` (N, S), ``labels`` (N,) and scalar ``forcing_period`` (NaN when autonomous).
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from case_studies import PROJECT_ROOT
from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE
from case_studies.friction.friction_jax_ode import FrictionJaxODE
from case_studies.lorenz.lorenz_jax_ode import LorenzJaxODE
from case_studies.lorenz.setup_lorenz_system import lorenz_stop_event
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE

from pybasin.solvers import JaxSolver
from pybasin.solvers.jax_ode_system import JaxODESystem

DATA_DIR = Path(__file__).resolve().parent / "data"
GT_DIR = PROJECT_ROOT / "tests" / "integration"
SAMPLES_PER_FORCING_PERIOD = 64
DUFFING_N_PERIODS = 12


@dataclass(frozen=True)
class SystemSpec:
    name: str
    ode: JaxODESystem
    csv: Path
    t_end: float
    save_window: tuple[float, float]
    t_steps: int
    forcing_period: float
    drop_labels: tuple[str, ...] = ()
    use_stop_event: bool = False


def _duffing_window() -> tuple[tuple[float, float], int]:
    period = 2 * math.pi
    n_end = math.floor(1000.0 / period)
    start = (n_end - DUFFING_N_PERIODS) * period
    end = n_end * period
    return (start, end), DUFFING_N_PERIODS * SAMPLES_PER_FORCING_PERIOD + 1


DUFFING_WINDOW, DUFFING_STEPS = _duffing_window()

SYSTEMS = [
    SystemSpec(
        name="duffing",
        ode=DuffingJaxODE({"delta": 0.08, "k3": 1, "A": 0.2}),
        csv=GT_DIR / "duffing/ground_truths/main/main_duffing.csv",
        t_end=1000.0,
        save_window=DUFFING_WINDOW,
        t_steps=DUFFING_STEPS,
        forcing_period=2 * math.pi,
    ),
    SystemSpec(
        name="pendulum",
        ode=PendulumJaxODE({"alpha": 0.1, "T": 0.5, "K": 1.0}),
        csv=GT_DIR / "pendulum/ground_truths/case1/main_pendulum_case1.csv",
        t_end=1000.0,
        save_window=(950.0, 1000.0),
        t_steps=1000,
        forcing_period=float("nan"),
    ),
    SystemSpec(
        name="friction",
        ode=FrictionJaxODE({
            "v_d": 1.5,
            "xi": 0.05,
            "musd": 2.0,
            "mud": 0.5,
            "muv": 0.0,
            "v0": 0.5,
        }),
        csv=GT_DIR / "friction/ground_truths/main/main_friction.csv",
        t_end=500.0,
        save_window=(400.0, 500.0),
        t_steps=1000,
        forcing_period=float("nan"),
    ),
    SystemSpec(
        name="lorenz",
        ode=LorenzJaxODE({"sigma": 0.12, "r": 0.0, "b": -0.6}),
        csv=GT_DIR / "lorenz/ground_truths/main/main_lorenz.csv",
        t_end=1000.0,
        save_window=(900.0, 1000.0),
        t_steps=1000,
        forcing_period=float("nan"),
        drop_labels=("unbounded",),
        use_stop_event=True,
    ),
]


def generate(spec: SystemSpec) -> None:
    df = pd.read_csv(spec.csv)
    keep = ~df["label"].isin(spec.drop_labels).to_numpy()
    coord_cols = [c for c in df.columns if c != "label"]
    y0 = torch.tensor(np.asarray(df[coord_cols], dtype=np.float32)[keep])

    solver = JaxSolver(
        t_span=(0.0, spec.t_end),
        t_steps=spec.t_steps,
        t_eval=spec.save_window,
        device="cpu",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,
        event_fn=lorenz_stop_event if spec.use_stop_event else None,
    )
    start = time.perf_counter()
    t, y = solver.integrate(spec.ode, y0)
    elapsed = time.perf_counter() - start

    y_np = y.cpu().numpy().astype(np.float32)
    t_np = t.cpu().numpy()
    if not math.isnan(spec.forcing_period):
        y_np, t_np = y_np[..., :-1], t_np[:-1]
    finite = np.asarray(np.isfinite(y_np).all(axis=(1, 2)))
    if spec.use_stop_event:
        finite &= np.abs(y_np).max(axis=(1, 2)) < 1e3
    labels = np.asarray(df["label"], dtype=str)[keep]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / f"{spec.name}.npz",
        y=y_np[finite],
        t=t_np,
        y0=y0.numpy()[finite],
        labels=labels[finite],
        forcing_period=spec.forcing_period,
    )
    counts = pd.Series(labels[finite]).value_counts().to_dict()
    print(
        f"{spec.name}: {finite.sum()}/{len(finite)} kept, shape={y_np[finite].shape}, "
        f"{elapsed:.1f}s, classes={counts}"
    )


if __name__ == "__main__":
    for spec in SYSTEMS:
        generate(spec)
