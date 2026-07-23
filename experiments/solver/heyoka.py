# pyright: basic
"""Benchmark and correctness validation for the pendulum system using heyoka.py.

Uses heyoka's taylor_adaptive_batch (SIMD batch mode) combined with
ensemble_propagate_until_batch (multi-threading) for maximum throughput.
Each thread integrates BATCH_SIZE trajectories simultaneously via SIMD, so total
parallelism = N_THREADS × BATCH_SIZE.

Scaling benchmark matches the structure of src/zigode/benchmark_zig_solver.py.

Correctness is validated by computing basin stability and comparing to the JAX reference
pipeline defined in case_studies/pendulum/setup_pendulum_system.py.

Pendulum ODE:
    dθ/dt  = ω
    dω/dt  = −α·ω + T − K·sin(θ)

Parameters: alpha=0.1, T=0.5, K=1.0

Basin stability approximation (N=1000):
  - Random ICs sampled from the pendulum phase space
  - Feature: log_delta(ω) over the steady-state window [T_STEADY, T_SPAN[1]]
  - KNN (k=1) classifier trained on template trajectories for FP and LC attractors
  - Result: fraction of ICs converging to each attractor

Usage:
    uv run python -m experiments.solver.heyoka
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from statistics import mean, stdev

import heyoka as hy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Constants matching case_studies/pendulum/setup_pendulum_system.py
# ---------------------------------------------------------------------------
ALPHA: float = 0.1
T_PARAM: float = 0.5
K_PARAM: float = 1.0

T_SPAN: tuple[float, float] = (0.0, 1000.0)
T_STEADY: float = 950.0
N_STEADY_STEPS: int = 100

N_VALUES: list[int] = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
NUM_ROUNDS: int = 5
SEED: int = 42

TEMPLATE_Y0: list[list[float]] = [[0.4, 0.0], [2.7, 0.0]]
LABELS: list[str] = ["FP", "LC"]

RESULTS_DIR: Path = Path(__file__).parent / "results"

BATCH_SIZE: int = hy.recommended_simd_size()
N_THREADS: int = os.cpu_count() or 1


# ---------------------------------------------------------------------------
# ODE definition
# ---------------------------------------------------------------------------


def _make_eqns() -> list[tuple[hy.expression, hy.expression]]:
    """Return heyoka symbolic equations for the pendulum.

    Runtime parameters:
        par[0] = alpha (damping)
        par[1] = T     (external torque)
        par[2] = K     (stiffness)
    """
    theta, omega = hy.make_vars("theta", "omega")
    return [
        (theta, omega),
        (omega, -hy.par[0] * omega + hy.par[1] - hy.par[2] * hy.sin(theta)),
    ]


# ---------------------------------------------------------------------------
# IC generation
# ---------------------------------------------------------------------------


def _generate_ics(n: int) -> np.ndarray:
    """Generate *n* random ICs inside the pendulum sampling region."""
    theta_eq: float = float(np.arcsin(T_PARAM / K_PARAM))
    rng = np.random.default_rng(SEED)
    return np.column_stack([
        rng.uniform(-np.pi + theta_eq, np.pi + theta_eq, n),
        rng.uniform(-10.0, 10.0, n),
    ])


# ---------------------------------------------------------------------------
# Low-level batch helpers
# ---------------------------------------------------------------------------

_PARS_BLOCK: np.ndarray = np.array([
    [ALPHA] * BATCH_SIZE,
    [T_PARAM] * BATCH_SIZE,
    [K_PARAM] * BATCH_SIZE,
])


def _make_integrator(y0_batch: np.ndarray) -> hy.taylor_adaptive_batch:
    """Create a fresh batch integrator for *y0_batch* (shape [BATCH_SIZE, 2])."""
    return hy.taylor_adaptive_batch(
        _make_eqns(),
        state=y0_batch.T.copy(),  # shape (2, BATCH_SIZE)
        pars=_PARS_BLOCK.copy(),
    )


def _pad_ics(y0_all: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad *y0_all* to a multiple of BATCH_SIZE and return (padded, n_batches)."""
    n: int = y0_all.shape[0]
    n_batches: int = math.ceil(n / BATCH_SIZE)
    padded: np.ndarray = np.zeros((n_batches * BATCH_SIZE, 2))
    padded[:n] = y0_all
    return padded, n_batches


def _reset(ta: hy.taylor_adaptive_batch, y0_batch: np.ndarray) -> None:
    """Reset *ta* to time 0 and update initial conditions."""
    ta.set_time([0.0] * BATCH_SIZE)
    ta.state[:] = y0_batch.T


# ---------------------------------------------------------------------------
# Ensemble batch integration (SIMD × multi-threaded)
# ---------------------------------------------------------------------------


def _integrate_ensemble(y0_all: np.ndarray, ta: hy.taylor_adaptive_batch) -> None:
    """Integrate all ICs to T_SPAN[1] using ensemble_propagate_until_batch.

    Each iteration integrates BATCH_SIZE trajectories via SIMD; N_THREADS iterations
    run in parallel, giving N_THREADS × BATCH_SIZE concurrent trajectories.
    """
    padded, n_batches = _pad_ics(y0_all)

    def gen(ta_copy: hy.taylor_adaptive_batch, idx: int) -> hy.taylor_adaptive_batch:
        ta_copy.set_time(0.0)
        ta_copy.state[:] = padded[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].T
        ta_copy.pars[:] = _PARS_BLOCK
        return ta_copy

    hy.ensemble_propagate_until_batch(ta, T_SPAN[1], n_batches, gen)


# ---------------------------------------------------------------------------
# Scaling benchmark
# ---------------------------------------------------------------------------


def _benchmark_scaling() -> list[dict[str, object]]:
    """Run the scaling benchmark using ensemble_propagate_until_batch."""
    print(f"\n{'=' * 60}")
    print(f"  heyoka ensemble batch  (BATCH_SIZE={BATCH_SIZE}, N_THREADS={N_THREADS})")
    print(f"{'=' * 60}")

    warmup_y0: np.ndarray = _generate_ics(BATCH_SIZE)
    ta_warmup: hy.taylor_adaptive_batch = _make_integrator(warmup_y0)
    _integrate_ensemble(warmup_y0, ta_warmup)

    results: list[dict[str, object]] = []
    for n in N_VALUES:
        y0: np.ndarray = _generate_ics(n)
        ta: hy.taylor_adaptive_batch = _make_integrator(y0[:BATCH_SIZE])
        round_times: list[float] = []

        for _ in range(NUM_ROUNDS):
            start: float = time.perf_counter()
            _integrate_ensemble(y0, ta)
            elapsed: float = time.perf_counter() - start
            round_times.append(elapsed)

        min_t: float = min(round_times)
        mean_t: float = mean(round_times)
        std_t: float = stdev(round_times) if len(round_times) > 1 else 0.0
        per_ic_us: float = min_t / n * 1e6

        print(
            f"  N={n:>7,}  min={min_t:.3f}s  mean={mean_t:.3f}s "
            f"+-{std_t:.3f}s  {per_ic_us:.1f} us/IC"
        )
        results.append({
            "N": n,
            "round_times": round_times,
            "mean_time": mean_t,
            "std_time": std_t,
            "min_time": min_t,
            "max_time": max(round_times),
        })
    return results


# ---------------------------------------------------------------------------
# Feature extraction  (ensemble_propagate_grid_batch)
# ---------------------------------------------------------------------------


def _log_delta(series: np.ndarray) -> float:
    """log(|max - mean| + 1e-10) over a 1-D time series."""
    return float(np.log(np.abs(series.max() - series.mean()) + 1e-10))


def _extract_features_batch(y0_all: np.ndarray) -> np.ndarray:
    """Return feature vector (log_delta of ω) for each IC in *y0_all*.

    Integrates each batch sequentially:
      1. propagate_until(T_STEADY) to skip the transient
      2. propagate_grid over [T_STEADY, T_SPAN[1]] to obtain the steady-state ω series
    Then computes log_delta per trajectory.
    """
    n: int = y0_all.shape[0]
    features: np.ndarray = np.empty(n)

    t_grid_1d: np.ndarray = np.linspace(T_STEADY, T_SPAN[1], N_STEADY_STEPS)
    t_grid: np.ndarray = np.repeat(t_grid_1d, BATCH_SIZE).reshape(-1, BATCH_SIZE)
    ends_steady: list[float] = [T_STEADY] * BATCH_SIZE

    padded, n_batches = _pad_ics(y0_all)
    ta: hy.taylor_adaptive_batch = _make_integrator(padded[:BATCH_SIZE])

    for i in range(n_batches):
        _reset(ta, padded[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])
        ta.propagate_until(ends_steady)
        _, out = ta.propagate_grid(t_grid)
        # out shape: (n_t, n_states, BATCH_SIZE)
        omega_series: np.ndarray = out[:, 1, :]
        base: int = i * BATCH_SIZE
        count: int = min(BATCH_SIZE, n - base)
        for b in range(count):
            features[base + b] = _log_delta(omega_series[:, b])

    return features


# ---------------------------------------------------------------------------
# Basin stability
# ---------------------------------------------------------------------------

N_BASIN: int = 1000


def _compute_basin_stability() -> dict[str, object]:
    """Compute basin stability for the pendulum using heyoka batch integration."""
    print(f"\n{'=' * 60}")
    print(f"  Basin stability validation  (N={N_BASIN})")
    print(f"{'=' * 60}")

    # --- Template features (one IC per attractor) ---
    template_y0: np.ndarray = np.array(TEMPLATE_Y0, dtype=float)
    template_features: np.ndarray = _extract_features_batch(template_y0).reshape(-1, 1)

    knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=1)
    knn.fit(template_features, LABELS)

    # --- Sample ICs ---
    y0_all: np.ndarray = _generate_ics(N_BASIN)
    sample_features: np.ndarray = _extract_features_batch(y0_all).reshape(-1, 1)
    predictions: np.ndarray = np.array(knn.predict(sample_features))

    basin_stability: dict[str, float] = {
        label: float(np.mean(predictions == label)) for label in LABELS
    }

    print("  Basin stability estimates:")
    for label, bs in basin_stability.items():
        print(f"    {label}: {bs:.4f}")

    return {
        "n": N_BASIN,
        "basin_stability": basin_stability,
        "template_features": {
            LABELS[i]: float(template_features[i, 0]) for i in range(len(LABELS))
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"heyoka version: {hy.__version__}")
    print(f"Recommended SIMD batch size: {BATCH_SIZE}")
    print(f"CPU threads: {N_THREADS}")
    print(f"Effective parallelism: {BATCH_SIZE * N_THREADS} trajectories/dispatch")
    print(f"N values: {N_VALUES}")
    print(f"Rounds: {NUM_ROUNDS}  |  t_span: {T_SPAN}")

    benchmark_results: list[dict[str, object]] = _benchmark_scaling()
    basin_result: dict[str, object] = _compute_basin_stability()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bench_path: Path = RESULTS_DIR / "heyoka_scaling.json"
    with open(bench_path, "w") as f:
        json.dump(
            {
                "heyoka_version": str(hy.__version__),
                "batch_size": BATCH_SIZE,
                "n_threads": N_THREADS,
                "num_rounds": NUM_ROUNDS,
                "t_span": list(T_SPAN),
                "benchmarks": benchmark_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {bench_path}")

    basin_path: Path = RESULTS_DIR / "heyoka_basin_stability.json"
    with open(basin_path, "w") as f:
        json.dump(basin_result, f, indent=2)
    print(f"Saved: {basin_path}")


if __name__ == "__main__":
    main()
