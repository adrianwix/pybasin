"""Benchmark the Zig Dopri5 solver on the pendulum system (Zig ODE vs SymPy C ODE).

Scaling benchmark across multiple N values, matching the pynamicalsys benchmark structure.
Region of interest matches case_studies/pendulum/setup_pendulum_system.py:
  theta: [-pi + arcsin(T/K), pi + arcsin(T/K)]
  omega: [-10, 10]

Usage:
    uv run python -m zigode.benchmark_zig_solver
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import sympy as sp

from zigode.sympy_ode import SymPyODE
from zigode.zig_solver import ODEDefinition, ZigODE, ZigODESolver

N_VALUES: list[int] = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
NUM_ROUNDS: int = 5
N_STEPS: int = 10_000
T_SPAN: tuple[float, float] = (0.0, 1000.0)
SEED: int = 42

PARAMS: dict[str, float] = {"alpha": 0.1, "T": 0.5, "K": 1.0}

RESULTS_DIR: Path = Path(__file__).parent / "results"

PENDULUM_ZIG: ZigODE = ZigODE(name="pendulum", param_names=["alpha", "T", "K"])


def _make_sympy_pendulum() -> SymPyODE:
    """Create a SymPy pendulum ODE (identical RHS to ``pendulum.zig``)."""
    theta, dtheta = sp.symbols("theta dtheta")  # type: ignore[reportUnknownMemberType]
    alpha, T, K = sp.symbols("alpha T K")  # type: ignore[reportUnknownMemberType]
    return SymPyODE(
        name="pendulum_sympy",
        state=[theta, dtheta],
        params=[alpha, T, K],
        rhs=[dtheta, -alpha * dtheta + T - K * sp.sin(theta)],
    )


def _generate_ics(n: int) -> np.ndarray:
    """Generate n random initial conditions in the pendulum sampling region."""
    theta_eq: float = float(np.arcsin(PARAMS["T"] / PARAMS["K"]))
    rng = np.random.default_rng(SEED)
    return np.column_stack([
        rng.uniform(-np.pi + theta_eq, np.pi + theta_eq, n),
        rng.uniform(-10.0, 10.0, n),
    ])


def _benchmark_ode_scaling(
    label: str,
    ode: ODEDefinition,
    solver: ZigODESolver,
    t_eval: np.ndarray,
) -> list[dict[str, object]]:
    """Run scaling benchmark for one ODE across all N values."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    # Warmup
    solver.solve(ode, _generate_ics(10), T_SPAN, t_eval, PARAMS)

    results: list[dict[str, object]] = []
    for n in N_VALUES:
        y0: np.ndarray = _generate_ics(n)
        round_times: list[float] = []

        for _ in range(1, NUM_ROUNDS + 1):
            start: float = time.perf_counter()
            solver.solve(ode, y0, T_SPAN, t_eval, PARAMS)
            elapsed: float = time.perf_counter() - start
            round_times.append(elapsed)

        mean_t: float = mean(round_times)
        std_t: float = stdev(round_times) if len(round_times) > 1 else 0.0
        min_t: float = min(round_times)
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


def main() -> None:
    pendulum_sympy: SymPyODE = _make_sympy_pendulum()
    solver = ZigODESolver()

    t_eval: np.ndarray = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)

    print(f"N values: {N_VALUES}")
    print(f"Rounds: {NUM_ROUNDS}  |  Steps: {N_STEPS:,}  |  t_span: {T_SPAN}")

    zig_results: list[dict[str, object]] = _benchmark_ode_scaling(
        "Zig ODE (pendulum.zig)", PENDULUM_ZIG, solver, t_eval
    )
    sympy_results: list[dict[str, object]] = _benchmark_ode_scaling(
        "SymPy ODE (pendulum_sympy.c)", pendulum_sympy, solver, t_eval
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, data in [("zig", zig_results), ("sympy_c", sympy_results)]:
        path: Path = RESULTS_DIR / f"zig_solver_{name}_scaling.json"
        with open(path, "w") as f:
            json.dump(
                {"num_rounds": NUM_ROUNDS, "n_steps": N_STEPS, "benchmarks": data}, f, indent=2
            )
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
