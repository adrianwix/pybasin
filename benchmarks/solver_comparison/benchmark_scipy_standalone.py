# pyright: basic
"""
Standalone benchmark for scipy solve_ivp (DOP853) on the pendulum system.

No pytest-benchmark dependency. Runs each N value for ROUNDS repetitions,
prints a summary table, and writes results to a JSON file.

Run with:
    uv run python -m benchmarks.solver_comparison.benchmark_scipy_standalone
"""

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp

# ── Configuration ─────────────────────────────────────────────────────────────

ROUNDS = 5
N_VALUES: list[int] = [100, 200, 500, 1_000, 2_000, 5_000, 10_000]

ALPHA = 0.1
T = 0.5
K = 1.0
TIME_SPAN = (0.0, 1000.0)
N_STEPS = 1000
RTOL = 1e-8
ATOL = 1e-6

MIN_LIMITS = [-np.pi + np.arcsin(T / K), -10.0]
MAX_LIMITS = [np.pi + np.arcsin(T / K), 10.0]

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = RESULTS_DIR / "python_scipy_benchmark_results.json"

# ── ODE ───────────────────────────────────────────────────────────────────────


def ode_scipy(t: float, y: np.ndarray) -> np.ndarray:
    theta, omega = y
    return np.array([omega, -ALPHA * omega + T - K * np.sin(theta)])


# ── Solver ────────────────────────────────────────────────────────────────────


def run_scipy(y0: np.ndarray) -> np.ndarray:
    t_eval = np.linspace(TIME_SPAN[0], TIME_SPAN[1], N_STEPS)
    n = y0.shape[0]

    def solve_single(y0_single: np.ndarray) -> np.ndarray:
        sol = solve_ivp(
            ode_scipy,
            TIME_SPAN,
            y0_single,
            method="DOP853",
            t_eval=t_eval,
            rtol=RTOL,
            atol=ATOL,
        )
        return sol.y.T  # type: ignore[no-any-return]

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(solve_single)(y0[i])
        for i in range(n)  # type: ignore[misc]
    )
    return np.stack(results, axis=1)  # type: ignore[arg-type]


# ── Benchmark loop ────────────────────────────────────────────────────────────


def benchmark_n(n: int) -> dict[str, object]:
    rng = np.random.default_rng(42)
    y0 = np.empty((n, 2), dtype=np.float64)
    y0[:, 0] = rng.uniform(MIN_LIMITS[0], MAX_LIMITS[0], n)
    y0[:, 1] = rng.uniform(MIN_LIMITS[1], MAX_LIMITS[1], n)

    times: list[float] = []
    for r in range(1, ROUNDS + 1):
        t_start = time.perf_counter()
        result = run_scipy(y0)
        elapsed = time.perf_counter() - t_start
        times.append(elapsed)
        print(f"  n={n:>6}  round {r}/{ROUNDS}  {elapsed:.3f}s   shape={result.shape}")

    mean = float(np.mean(times))
    std = float(np.std(times))
    return {
        "n": n,
        "rounds": ROUNDS,
        "times": times,
        "mean": mean,
        "std": std,
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("scipy DOP853 standalone benchmark")
    print(f"Rounds per N : {ROUNDS}")
    print(f"N values     : {N_VALUES}")
    print(f"Time span    : {TIME_SPAN}  |  n_steps={N_STEPS}")
    print(f"Tolerances   : rtol={RTOL}  atol={ATOL}")
    print("=" * 60)

    all_results: list[dict[str, object]] = []
    for n in N_VALUES:
        print(f"\n-- N = {n} --")
        result = benchmark_n(n)
        all_results.append(result)

    print("\n" + "=" * 60)
    print(f"{'N':>8}  {'mean (s)':>10}  {'std (s)':>9}  {'min (s)':>9}  {'max (s)':>9}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['n']:>8}  {r['mean']:>10.3f}  {r['std']:>9.3f}  {r['min']:>9.3f}  {r['max']:>9.3f}"
        )

    OUTPUT_FILE.write_text(json.dumps({"benchmarks": all_results}, indent=2))
    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
