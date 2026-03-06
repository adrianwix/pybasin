# pyright: basic
"""
Experiment: Tail-only trajectory saving and its interaction with time_steady.

Integration stores every time point by default. By saving only the tail
portion (e.g., t ∈ [800, 1000] of a trajectory integrated over t ∈ [0, 1000]),
we reduce memory by 80% while retaining everything needed for steady-state
feature extraction.

Questions under test:
  1. Diffrax SaveAt — can we save only t ∈ [800, 1000] after integrating from 0?
  2. torchdiffeq — does it have an equivalent mechanism?
  3. time_steady interaction — what happens when time_steady is:
       700  → before saved range  (all 200 saved points pass the filter)
       900  → within saved range  (only t ∈ [900, 1000] pass)
      1100  → beyond saved range  (no points pass → fallback to all 200)

Expected result: features from the tail-only save should match those from a full
save after applying the same time_steady filter, because the pendulum is in
steady-state by t=800.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchode as to  # type: ignore[import-untyped]
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from torchdiffeq import odeint  # type: ignore[import-untyped]

from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from case_studies.pendulum.pendulum_ode import PendulumNumpyODE, PendulumODE
from pybasin.jax_utils import jax_to_torch
from pybasin.solution import Solution
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

PARAMS: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}

N_BATCH = 200
T_START = 0.0
T_END = 1000.0
SAVE_START = 800.0  # Only save the trajectory from here onwards
N_SAVE = 200  # Number of time points in [800, 1000]
N_FULL = 1000  # Number of time points for the full [0, 1000] baseline

RTOL = 1e-8
ATOL = 1e-6


def make_y0(n: int) -> torch.Tensor:
    rng = np.random.default_rng(42)
    theta = rng.uniform(
        -np.pi + np.arcsin(PARAMS["T"] / PARAMS["K"]),
        np.pi + np.arcsin(PARAMS["T"] / PARAMS["K"]),
        n,
    )
    theta_dot = rng.uniform(-10.0, 10.0, n)
    return torch.tensor(np.stack([theta, theta_dot], axis=1), dtype=torch.float64)


FEATURE_CONFIG = {1: {"maximum": None, "minimum": None, "mean": None}}


def make_solution(t: torch.Tensor, y: torch.Tensor, y0: torch.Tensor) -> Solution:
    # y shape from torchdiffeq: (n_steps, batch, dims)
    # y shape from diffrax+jax_to_torch: same convention after transpose in JaxSolver
    return Solution(initial_condition=y0, time=t, y=y)


# ── helpers ────────────────────────────────────────────────────────────────────


def extract(solution: Solution, time_steady: float) -> torch.Tensor:
    """Run TorchFeatureExtractor on a Solution and return feature tensor."""
    extractor = TorchFeatureExtractor(
        time_steady=time_steady,
        features=None,
        features_per_state=FEATURE_CONFIG,
        normalize=False,
    )
    return extractor.extract_features(solution)


def summarise_filter(solution: Solution, time_steady: float) -> dict[str, object]:
    """Return how many time points survive the time_steady filter."""
    time_arr = solution.time
    survivors = int((time_arr >= time_steady).sum().item())
    total = int(time_arr.shape[0])
    return {
        "time_steady": time_steady,
        "total_saved_points": total,
        "points_after_filter": survivors,
        "t_min": float(time_arr[0]),
        "t_max": float(time_arr[-1]),
    }


# ── Diffrax experiments ─────────────────────────────────────────────────────────


def run_diffrax_full_save(y0_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: integrate and save at every point in [0, 1000]."""
    ode = PendulumJaxODE(PARAMS)
    y0_jax = jnp.array(y0_np)

    def ode_fn(t, y, args):  # type: ignore[no-untyped-def]
        return ode.ode(t, y)

    t_eval = jnp.linspace(T_START, T_END, N_FULL)

    def solve_single(y0_single):  # type: ignore[no-untyped-def]
        sol = diffeqsolve(
            ODETerm(ode_fn),
            Dopri5(),
            t0=T_START,
            t1=T_END,
            dt0=None,
            y0=y0_single,
            saveat=SaveAt(ts=t_eval),
            stepsize_controller=PIDController(rtol=RTOL, atol=ATOL),
            max_steps=16**5,
        )
        return sol.ys

    y_batch = jax.vmap(solve_single)(y0_jax)
    jax.block_until_ready(y_batch)  # type: ignore[no-untyped-call]

    # y_batch: (batch, n_steps, dims) → transpose to (n_steps, batch, dims)
    y_transposed = jnp.transpose(y_batch, (1, 0, 2))
    return jax_to_torch(t_eval, "cpu"), jax_to_torch(y_transposed, "cpu")


def run_diffrax_tail_save(y0_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-only: integrate over [0, 1000] but save only at t ∈ [800, 1000].

    Uses SaveAt(t0=True, ts=...) where t0=True additionally saves the very
    first point (t=0). We include it here purely to illustrate the t0 option;
    in practice you would omit t0=True to save only the tail.
    """
    ode = PendulumJaxODE(PARAMS)
    y0_jax = jnp.array(y0_np)

    def ode_fn(t, y, args):  # type: ignore[no-untyped-def]
        return ode.ode(t, y)

    # Save at t=0 (the initial point) plus the entire tail [800, 1000].
    # Diffrax SaveAt(t0=True) records the state at exactly t0 of the solve.
    ts_tail = jnp.linspace(SAVE_START, T_END, N_SAVE)

    def solve_single(y0_single):  # type: ignore[no-untyped-def]
        sol = diffeqsolve(
            ODETerm(ode_fn),
            Dopri5(),
            t0=T_START,
            t1=T_END,
            dt0=None,
            y0=y0_single,
            saveat=SaveAt(t0=True, ts=ts_tail),
            stepsize_controller=PIDController(rtol=RTOL, atol=ATOL),
            max_steps=16**5,
        )
        return sol.ts, sol.ys

    ts_batch, y_batch = jax.vmap(solve_single)(y0_jax)
    jax.block_until_ready(y_batch)  # type: ignore[no-untyped-call]

    # sol.ts: (batch, n_saved+1) — all rows equal, take first
    t_saved = ts_batch[0]
    # y_batch: (batch, n_saved+1, dims) → (n_saved+1, batch, dims)
    y_transposed = jnp.transpose(y_batch, (1, 0, 2))
    return jax_to_torch(t_saved, "cpu"), jax_to_torch(y_transposed, "cpu")


def run_diffrax_tail_only(y0_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-only without t0: integrate over [0, 1000], save only t ∈ [800, 1000].

    This is the memory-efficient variant — no initial state is stored.
    """
    ode = PendulumJaxODE(PARAMS)
    y0_jax = jnp.array(y0_np)

    def ode_fn(t, y, args):  # type: ignore[no-untyped-def]
        return ode.ode(t, y)

    ts_tail = jnp.linspace(SAVE_START, T_END, N_SAVE)

    def solve_single(y0_single):  # type: ignore[no-untyped-def]
        sol = diffeqsolve(
            ODETerm(ode_fn),
            Dopri5(),
            t0=T_START,
            t1=T_END,
            dt0=None,
            y0=y0_single,
            saveat=SaveAt(ts=ts_tail),
            stepsize_controller=PIDController(rtol=RTOL, atol=ATOL),
            max_steps=16**5,
        )
        return sol.ts, sol.ys

    ts_batch, y_batch = jax.vmap(solve_single)(y0_jax)
    jax.block_until_ready(y_batch)  # type: ignore[no-untyped-call]

    t_saved = ts_batch[0]
    y_transposed = jnp.transpose(y_batch, (1, 0, 2))
    return jax_to_torch(t_saved, "cpu"), jax_to_torch(y_transposed, "cpu")


# ── torchdiffeq experiments ────────────────────────────────────────────────────


def run_torchdiffeq_full_save(y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: integrate and save at every point in [0, 1000]."""
    ode = PendulumODE(PARAMS)
    t_eval = torch.linspace(T_START, T_END, N_FULL, dtype=torch.float64)
    with torch.no_grad():
        y = odeint(ode, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL)  # type: ignore[arg-type]
    return t_eval, y


def run_torchdiffeq_tail_save(y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-only equivalent for torchdiffeq.

    torchdiffeq does not have a dedicated SaveAt mechanism. Instead, we pass a
    time array that starts at T_START (t=0) so the solver integrates from the
    correct initial conditions, and then jumps directly to SAVE_START (t=800)
    with fine steps through to T_END (t=1000).

    The solver will integrate across the [0, 800] gap with its adaptive stepper
    (no dense output stored) and only evaluate/store the solution at the listed
    time points. We keep the t=0 entry in the output so we can verify y0 is
    reproduced exactly, then drop it before passing the data to the feature
    extractor.
    """
    ode = PendulumODE(PARAMS)

    # One anchor point at T_START ensures integration begins at the correct IC.
    ts_tail = torch.linspace(SAVE_START, T_END, N_SAVE, dtype=torch.float64)
    t_eval = torch.cat([torch.tensor([T_START], dtype=torch.float64), ts_tail])

    with torch.no_grad():
        y_all = odeint(ode, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL)  # type: ignore[arg-type]

    # Drop the t=0 state — it is identical to y0 and not needed for features.
    t_tail = t_eval[1:]
    y_tail = y_all[1:, ...]
    return t_tail, y_tail


def run_torchdiffeq_tail_no_anchor(y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-only without the T_START anchor point.

    torchdiffeq treats t_eval[0] as the initial time of the integration.
    Omitting T_START means the solver starts at t=800, treating y0
    (the state at t=0) as if it were the state at t=800. The trajectory
    over [800, 1000] is therefore wrong.

    However, for a dissipative system like the pendulum that has already
    converged to an attractor by t=800, the steady-state features over
    [900, 1000] may still look similar to the correct answer — not because
    the approach is right, but because both trajectories are on the same
    attractor regardless of where they started.
    """
    ode = PendulumODE(PARAMS)
    t_eval = torch.linspace(SAVE_START, T_END, N_SAVE, dtype=torch.float64)
    with torch.no_grad():
        y = odeint(ode, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL)  # type: ignore[arg-type]
    return t_eval, y


# ── torchode experiments ───────────────────────────────────────────────────────


def _torchode_solve(
    y0: torch.Tensor,
    t_eval: torch.Tensor,
    t_start_val: float,
    t_end_val: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared helper: run torchode with explicit t_start/t_end and given t_eval."""
    batch_size = y0.shape[0]
    dtype = y0.dtype
    device = y0.device

    def ode_fn(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ode = PendulumODE(PARAMS)
        return ode(t, y)

    term = to.ODETerm(ode_fn)  # pyright: ignore[reportArgumentType]
    step_method = to.Dopri5(term=term)
    controller = to.IntegralController(atol=ATOL, rtol=RTOL, term=term)
    adjoint = to.AutoDiffAdjoint(step_method, controller)

    t_start = torch.full((batch_size,), t_start_val, dtype=dtype, device=device)
    t_end = torch.full((batch_size,), t_end_val, dtype=dtype, device=device)
    t_eval_batched = t_eval.unsqueeze(0).expand(batch_size, -1)

    problem = to.InitialValueProblem(y0=y0, t_start=t_start, t_end=t_end, t_eval=t_eval_batched)
    with torch.no_grad():
        sol = adjoint.solve(problem)

    # sol.ys: (batch, n_eval, dims) → (n_eval, batch, dims)
    y_out: torch.Tensor = sol.ys.permute(1, 0, 2)  # type: ignore[union-attr]
    return t_eval, y_out


def run_torchode_full_save(y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: integrate [0, 1000] and save all 1000 points."""
    t_eval = torch.linspace(T_START, T_END, N_FULL, dtype=y0.dtype)
    return _torchode_solve(y0, t_eval, T_START, T_END)


def run_torchode_tail_save(y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-only: integrate [0, 1000] but save only t ∈ [800, 1000].

    torchode's InitialValueProblem accepts separate t_start/t_end and t_eval
    arguments. t_start and t_end define the integration interval; t_eval defines
    where the solution is stored. Setting t_start=0, t_end=1000, and
    t_eval=linspace(800, 1000, 200) integrates the full span correctly while
    materialising only the tail — no anchor trick required, unlike torchdiffeq.
    """
    t_eval = torch.linspace(SAVE_START, T_END, N_SAVE, dtype=y0.dtype)
    return _torchode_solve(y0, t_eval, T_START, T_END)


# ── scipy experiments ─────────────────────────────────────────────────────────


def _scipy_solve(
    y0_np: np.ndarray,
    t_eval_np: np.ndarray,
    t_span: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared helper: solve a batch of IVPs with scipy's DOP853."""
    ode = PendulumNumpyODE(PARAMS)

    def solve_single(y0_single: np.ndarray) -> np.ndarray:
        sol = solve_ivp(  # type: ignore[no-untyped-call]
            fun=ode,
            t_span=t_span,
            y0=y0_single,
            method="DOP853",
            t_eval=t_eval_np,
            rtol=RTOL,
            atol=ATOL,
        )
        return sol.y.T  # type: ignore[no-any-return]

    results = Parallel(n_jobs=-1)(delayed(solve_single)(y0_np[i]) for i in range(y0_np.shape[0]))  # type: ignore[misc]
    y_np: np.ndarray = np.stack(results, axis=1)  # type: ignore[arg-type]
    t_torch = torch.tensor(t_eval_np, dtype=torch.float64)
    y_torch = torch.tensor(y_np, dtype=torch.float64)
    return t_torch, y_torch


def run_scipy_full_save(y0_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: integrate [0, 1000] and save all 1000 points."""
    t_eval_np = np.linspace(T_START, T_END, N_FULL)
    return _scipy_solve(y0_np, t_eval_np, (T_START, T_END))


def run_scipy_tail_save(y0_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Tail-only: integrate [0, 1000] but save only t ∈ [800, 1000].

    scipy's solve_ivp separates t_span (integration bounds) from t_eval
    (evaluation points). Passing t_span=(0, 1000) with
    t_eval=linspace(800, 1000, 200) integrates the full span from t=0 and
    stores only the tail.
    """
    t_eval_np = np.linspace(SAVE_START, T_END, N_SAVE)
    return _scipy_solve(y0_np, t_eval_np, (T_START, T_END))


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-scipy", action="store_true", help="Run only the scipy sections")
    args = parser.parse_args()
    only_scipy: bool = args.only_scipy

    y0_torch = make_y0(N_BATCH)
    y0_np = y0_torch.numpy()

    print("=" * 70)
    print("Pendulum parameters:", PARAMS)
    print(f"Batch size: {N_BATCH}  |  integration span: [{T_START}, {T_END}]")
    print(f"Tail region: [{SAVE_START}, {T_END}]  ({N_SAVE} pts)  vs full: {N_FULL} pts")
    print("=" * 70)

    if not only_scipy:
        # ── 1. Diffrax full save (baseline) ──────────────────────────────────
        print("\n── Diffrax: full save [0, 1000] (baseline) ─────────────────────────")
        t0 = time.perf_counter()
        t_full, y_full = run_diffrax_full_save(y0_np)
        elapsed = time.perf_counter() - t0
        sol_full = make_solution(t_full, y_full, y0_torch)
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(f"  t range  : [{float(t_full[0]):.0f}, {float(t_full[-1]):.0f}]")
        print(f"  y shape  : {tuple(y_full.shape)}  (n_steps, batch, dims)")
        print(f"  Memory   : ~{y_full.element_size() * y_full.numel() / 1e6:.1f} MB")

        # ── 2. Diffrax tail save with t0=True ────────────────────────────────
        print("\n── Diffrax: SaveAt(t0=True, ts=[800..1000]) ────────────────────────")
        t0 = time.perf_counter()
        t_tail_t0, y_tail_t0 = run_diffrax_tail_save(y0_np)
        elapsed = time.perf_counter() - t0
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(
            f"  t range  : [{float(t_tail_t0[0]):.1f}, {float(t_tail_t0[-1]):.1f}]  ({len(t_tail_t0)} pts — includes t0=0)"
        )
        print(f"  y shape  : {tuple(y_tail_t0.shape)}")
        print(f"  Memory   : ~{y_tail_t0.element_size() * y_tail_t0.numel() / 1e6:.1f} MB")
        print("  Note: t[0]=0 is the initial state saved by t0=True; t[1:] is the tail.")

        # ── 3. Diffrax tail-only (no t0) ─────────────────────────────────────
        print("\n── Diffrax: SaveAt(ts=[800..1000]) — tail only ─────────────────────")
        t0 = time.perf_counter()
        t_tail, y_tail = run_diffrax_tail_only(y0_np)
        elapsed = time.perf_counter() - t0
        sol_tail = make_solution(t_tail, y_tail, y0_torch)
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(
            f"  t range  : [{float(t_tail[0]):.0f}, {float(t_tail[-1]):.0f}]  ({len(t_tail)} pts)"
        )
        print(f"  y shape  : {tuple(y_tail.shape)}")
        print(f"  Memory   : ~{y_tail.element_size() * y_tail.numel() / 1e6:.1f} MB")

        # ── 4. torchdiffeq full save (baseline) ──────────────────────────────
        print("\n── torchdiffeq: full save [0, 1000] (baseline) ─────────────────────")
        t0 = time.perf_counter()
        t_td_full, y_td_full = run_torchdiffeq_full_save(y0_torch)
        elapsed = time.perf_counter() - t0
        sol_td_full = make_solution(t_td_full, y_td_full, y0_torch)
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(f"  t range  : [{float(t_td_full[0]):.0f}, {float(t_td_full[-1]):.0f}]")
        print(f"  y shape  : {tuple(y_td_full.shape)}")
        print(f"  Memory   : ~{y_td_full.element_size() * y_td_full.numel() / 1e6:.1f} MB")

        # ── 5. torchdiffeq tail save ──────────────────────────────────────────
        print("\n── torchdiffeq: t_eval=[0] + linspace(800, 1000, 200) ──────────────")
        print("  (integrates from t=0; saves only the tail; t=0 entry is dropped)")
        t0 = time.perf_counter()
        t_td_tail, y_td_tail = run_torchdiffeq_tail_save(y0_torch)
        elapsed = time.perf_counter() - t0
        sol_td_tail = make_solution(t_td_tail, y_td_tail, y0_torch)
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(
            f"  t range  : [{float(t_td_tail[0]):.0f}, {float(t_td_tail[-1]):.0f}]  ({len(t_td_tail)} pts)"
        )
        print(f"  y shape  : {tuple(y_td_tail.shape)}")
        print(f"  Memory   : ~{y_td_tail.element_size() * y_td_tail.numel() / 1e6:.1f} MB")

        # ── 5b. torchdiffeq tail save WITHOUT T_START anchor ─────────────────
        print("\n── torchdiffeq: t_eval=linspace(800, 1000, 200) — NO anchor ─────────")
        print("  (starts at t=800 with y0 as IC; conceptually wrong, but attractor")
        print("   convergence by t=800 makes features deceptively close to correct)")
        t0 = time.perf_counter()
        t_td_no_anchor, y_td_no_anchor = run_torchdiffeq_tail_no_anchor(y0_torch)
        elapsed = time.perf_counter() - t0
        sol_td_no_anchor = make_solution(t_td_no_anchor, y_td_no_anchor, y0_torch)
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(
            f"  t range  : [{float(t_td_no_anchor[0]):.0f}, {float(t_td_no_anchor[-1]):.0f}]  ({len(t_td_no_anchor)} pts)"
        )
        print(f"  y shape  : {tuple(y_td_no_anchor.shape)}")
        feat_no_anchor = extract(sol_td_no_anchor, 900.0)
        feat_with_anchor = extract(sol_td_tail, 900.0)
        diff_anchor = (feat_no_anchor - feat_with_anchor).abs().max().item()
        print(
            f"  max |Δfeature| vs anchored tail : {diff_anchor:.6e}  (small because system is already on attractor by t=800)"
        )

        # ── 5c. torchode full save (baseline) ────────────────────────────────
        print("\n── torchode: full save [0, 1000] (baseline) ─────────────────────────")
        t0 = time.perf_counter()
        t_to_full, y_to_full = run_torchode_full_save(y0_torch)
        elapsed = time.perf_counter() - t0
        sol_to_full = make_solution(t_to_full, y_to_full, y0_torch)
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(f"  t range  : [{float(t_to_full[0]):.0f}, {float(t_to_full[-1]):.0f}]")
        print(f"  y shape  : {tuple(y_to_full.shape)}")
        print(f"  Memory   : ~{y_to_full.element_size() * y_to_full.numel() / 1e6:.1f} MB")

        # ── 5d. torchode tail save (explicit t_start/t_end) ──────────────────
        print("\n── torchode: t_start=0, t_end=1000, t_eval=linspace(800,1000,200) ───")
        print("  (CORRECT: t_start/t_end separate from t_eval — no anchor needed)")
        t0 = time.perf_counter()
        t_to_tail, y_to_tail = run_torchode_tail_save(y0_torch)
        elapsed = time.perf_counter() - t0
        sol_to_tail = make_solution(t_to_tail, y_to_tail, y0_torch)
        print(f"  Elapsed  : {elapsed:.2f}s")
        print(
            f"  t range  : [{float(t_to_tail[0]):.0f}, {float(t_to_tail[-1]):.0f}]  ({len(t_to_tail)} pts)"
        )
        print(f"  y shape  : {tuple(y_to_tail.shape)}")
        print(f"  Memory   : ~{y_to_tail.element_size() * y_to_tail.numel() / 1e6:.1f} MB")
        feat_to_tail = extract(sol_to_tail, 900.0)
        feat_to_full = extract(sol_to_full, 900.0)
        diff_to = (feat_to_tail - feat_to_full).abs().max().item()
        print(f"  max |Δfeature| tail vs full : {diff_to:.6e}")

    # ── scipy full save (baseline) ────────────────────────────────────────────
    print("\n── scipy: full save [0, 1000] (baseline) ─────────────────────────────")
    t0 = time.perf_counter()
    t_sc_full, y_sc_full = run_scipy_full_save(y0_np)
    elapsed = time.perf_counter() - t0
    sol_sc_full = make_solution(t_sc_full, y_sc_full, y0_torch)
    print(f"  Elapsed  : {elapsed:.2f}s")
    print(f"  t range  : [{float(t_sc_full[0]):.0f}, {float(t_sc_full[-1]):.0f}]")
    print(f"  y shape  : {tuple(y_sc_full.shape)}")
    print(f"  Memory   : ~{y_sc_full.element_size() * y_sc_full.numel() / 1e6:.1f} MB")

    # ── scipy tail save (t_span vs t_eval separation) ────────────────────────
    print("\n── scipy: t_span=(0,1000), t_eval=linspace(800,1000,200) ────────────")
    print("  (t_span and t_eval are separate args)")
    t0 = time.perf_counter()
    t_sc_tail, y_sc_tail = run_scipy_tail_save(y0_np)
    elapsed = time.perf_counter() - t0
    sol_sc_tail = make_solution(t_sc_tail, y_sc_tail, y0_torch)
    print(f"  Elapsed  : {elapsed:.2f}s")
    print(
        f"  t range  : [{float(t_sc_tail[0]):.0f}, {float(t_sc_tail[-1]):.0f}]  ({len(t_sc_tail)} pts)"
    )
    print(f"  y shape  : {tuple(y_sc_tail.shape)}")
    print(f"  Memory   : ~{y_sc_tail.element_size() * y_sc_tail.numel() / 1e6:.1f} MB")
    feat_sc_tail = extract(sol_sc_tail, 900.0)
    feat_sc_full = extract(sol_sc_full, 900.0)
    diff_sc = (feat_sc_tail - feat_sc_full).abs().max().item()
    print(f"  max |Δfeature| tail vs full : {diff_sc:.6e}")

    if not only_scipy:
        # ── time_steady interaction ───────────────────────────────────────────
        print("\n── time_steady interaction on the tail-only diffrax solution ────────")
        for ts_val in [700.0, 900.0, 1100.0]:
            info = summarise_filter(sol_tail, ts_val)
            feats = extract(sol_tail, ts_val)
            print(
                f"  time_steady={ts_val:.0f}:  {info['points_after_filter']}/{info['total_saved_points']} pts pass"
                f"  |  features shape: {tuple(feats.shape)}"
                f"  |  sample: {feats[0].tolist()}"
            )

        # ── Cross-solver feature comparison ──────────────────────────────────
        print("\n── Feature comparison: do solvers agree on the tail-only features? ──")
        time_steady_ref = 900.0
        feat_diffrax_tail = extract(sol_tail, time_steady_ref)
        feat_td_tail = extract(sol_td_tail, time_steady_ref)
        feat_diffrax_full = extract(sol_full, time_steady_ref)
        feat_td_full = extract(sol_td_full, time_steady_ref)
        feat_torchode_tail = extract(sol_to_tail, time_steady_ref)
        feat_torchode_full = extract(sol_to_full, time_steady_ref)
        feat_scipy_tail = extract(sol_sc_tail, time_steady_ref)
        feat_scipy_full = extract(sol_sc_full, time_steady_ref)

        diff_jax = (feat_diffrax_tail - feat_diffrax_full).abs().max().item()
        diff_td = (feat_td_tail - feat_td_full).abs().max().item()
        diff_to_cmp = (feat_torchode_tail - feat_torchode_full).abs().max().item()
        diff_sc_cmp = (feat_scipy_tail - feat_scipy_full).abs().max().item()
        diff_cross = (feat_diffrax_tail - feat_td_tail).abs().max().item()
        diff_cross_to = (feat_diffrax_tail - feat_torchode_tail).abs().max().item()
        diff_cross_sc = (feat_diffrax_tail - feat_scipy_tail).abs().max().item()
        print(f"  time_steady = {time_steady_ref:.0f}")
        print(f"  diffrax     tail vs full     : max |Δfeature| = {diff_jax:.6e}")
        print(f"  torchdiffeq tail vs full     : max |Δfeature| = {diff_td:.6e}")
        print(f"  torchode    tail vs full     : max |Δfeature| = {diff_to_cmp:.6e}")
        print(f"  scipy       tail vs full     : max |Δfeature| = {diff_sc_cmp:.6e}")
        print(f"  diffrax(tail) vs torchdiffeq(tail) : max |Δfeature| = {diff_cross:.6e}")
        print(f"  diffrax(tail) vs torchode(tail)    : max |Δfeature| = {diff_cross_to:.6e}")
        print(f"  diffrax(tail) vs scipy(tail)       : max |Δfeature| = {diff_cross_sc:.6e}")

        # ── Memory summary ────────────────────────────────────────────────────
        print("\n── Memory summary ───────────────────────────────────────────────────")
        mem_full = y_full.element_size() * y_full.numel()
        mem_tail = y_tail.element_size() * y_tail.numel()
        print(f"  Full save  ({N_FULL} pts):  {mem_full / 1e6:.2f} MB")
        print(f"  Tail save  ({N_SAVE} pts):  {mem_tail / 1e6:.2f} MB")
        print(f"  Reduction : {100 * (1 - mem_tail / mem_full):.1f}%")
    print()


if __name__ == "__main__":
    main()
