"""Zig ODE Solver tests — numerical correctness against JAX/Diffrax and torchdiffeq.

Compares the Zig Dopri5 solver against the JAX Dopri5 solver (via Diffrax)
and torchdiffeq Dopri5 on the pendulum ODE with identical parameters and tolerances.
Also tests the SymPy-to-C codegen path produces results identical to the Zig ODE.

Run:
    cd pyBasinWorkspace
    uv run python -m zigode.test_zig_solver
"""

from __future__ import annotations

import sys
import time
from typing import cast

import numpy as np
import sympy as sp
import torch

from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE
from case_studies.pendulum.pendulum_jax_ode import PendulumParams as PendulumJaxParams
from case_studies.pendulum.pendulum_ode import PendulumODE
from case_studies.pendulum.pendulum_ode import PendulumParams as PendulumTorchParams
from pybasin.solvers import JaxSolver, TorchDiffEqSolver
from zigode.sympy_ode import SymPyODE
from zigode.zig_solver import ZigODE, ZigODESolver

PARAMS: dict[str, float] = {"alpha": 0.1, "T": 0.5, "K": 1.0}
T_SPAN: tuple[float, float] = (0.0, 1000.0)
N_STEPS = 1000
RTOL = 1e-8
ATOL = 1e-6

TIME_GRID_ATOL = 1e-10
CROSS_SOLVER_ATOL = 0.05
FIXED_POINT_ATOL = 1e-3
LIMIT_CYCLE_OMEGA_ATOL = 1e-2
EARLY_TIME_ATOL = 1e-3
SYMPY_VS_ZIG_ATOL = 5e-5

TEST_ICS: list[list[float]] = [
    [0.4, 0.0],
    [2.7, 0.0],
    [0.5, -5.0],
    [-2.0, 3.0],
    [1.0, 0.0],
]

PENDULUM_ZIG: ZigODE = ZigODE(name="pendulum", param_names=["alpha", "T", "K"])


def _make_sympy_pendulum(name: str = "pendulum_sympy_test") -> SymPyODE:
    """Create a SymPy pendulum ODE identical to the Zig one."""
    theta, dtheta = sp.symbols("theta dtheta")  # type: ignore[reportUnknownMemberType]
    alpha, T, K = sp.symbols("alpha T K")  # type: ignore[reportUnknownMemberType]
    return SymPyODE(
        name=name,
        state=[theta, dtheta],
        params=[alpha, T, K],
        rhs=[dtheta, -alpha * dtheta + T - K * sp.sin(theta)],
    )


PENDULUM_SYMPY: SymPyODE = _make_sympy_pendulum()

passed: int = 0
failed: int = 0


def report(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def solve_with_zig(y0s: np.ndarray, t_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    solver = ZigODESolver(auto_rebuild_solver=False)
    t, y = solver.solve(PENDULUM_ZIG, y0s, T_SPAN, t_eval, params=PARAMS, rtol=RTOL, atol=ATOL)
    return t, y


def solve_with_jax(y0s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ode = PendulumJaxODE(cast(PendulumJaxParams, PARAMS))
    solver = JaxSolver(
        t_span=T_SPAN,
        t_steps=N_STEPS,
        rtol=RTOL,
        atol=ATOL,
        cache_dir=None,
    )
    y0_torch = torch.from_numpy(y0s)  # type: ignore[reportUnknownMemberType]
    t_torch, y_torch = solver.integrate(ode, y0_torch)
    t_np: np.ndarray = t_torch.numpy()
    y_np: np.ndarray = y_torch.numpy()
    return t_np, y_np


def solve_with_torchdiffeq(
    y0s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ode = PendulumODE(cast(PendulumTorchParams, PARAMS))
    solver = TorchDiffEqSolver(
        t_span=T_SPAN,
        t_steps=N_STEPS,
        method="dopri5",
        rtol=RTOL,
        atol=ATOL,
        cache_dir=None,
    )
    y0_torch = torch.from_numpy(y0s).to(solver.device)  # type: ignore[reportUnknownMemberType]
    t_torch, y_torch = solver.integrate(ode, y0_torch)
    t_np: np.ndarray = t_torch.cpu().numpy()
    y_np: np.ndarray = y_torch.cpu().numpy()
    return t_np, y_np


def test_shapes() -> None:
    print("\n--- Shape tests ---")

    solver = ZigODESolver(auto_rebuild_solver=False)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)

    # Single IC (1-D input)
    t, y = solver.solve(PENDULUM_ZIG, [0.4, 0.0], T_SPAN, t_eval, params=PARAMS)
    report("single IC returns 3-D", y.ndim == 3, f"shape={y.shape}")
    report("single IC shape (1, N, 2)", y.shape == (1, N_STEPS, 2), f"shape={y.shape}")

    # Batch IC (2-D input)
    y0s = np.array(TEST_ICS, dtype=np.float64)
    t, y = solver.solve(PENDULUM_ZIG, y0s, T_SPAN, t_eval, params=PARAMS)
    report(
        "batch IC shape (5, N, 2)",
        y.shape == (len(TEST_ICS), N_STEPS, 2),
        f"shape={y.shape}",
    )

    # t_eval returned correctly
    report("t_eval length", len(t) == N_STEPS, f"len={len(t)}")
    report("t_eval endpoints", bool(np.isclose(t[0], 0.0) and np.isclose(t[-1], 1000.0)))  # type: ignore[reportUnknownArgumentType]


def _compare_solver_pair(
    label: str,
    y_zig_t: np.ndarray,
    y_ref: np.ndarray,
    t_zig: np.ndarray,
    t_ref: np.ndarray,
) -> None:
    """Compare Zig results (n_steps, batch, dim) against a reference solver."""
    report(f"[{label}] time grids match", np.allclose(t_zig, t_ref, atol=TIME_GRID_ATOL))

    print(
        f"\n  {'IC':>20} {'Max |err|':>12} {'Max rel err':>12} "
        f"{'Zig final θ':>14} {f'{label} final θ':>14}"
    )
    print("  " + "-" * 80)

    overall_max_abs: float = 0.0
    for i, ic in enumerate(TEST_ICS):
        zig_traj: np.ndarray = y_zig_t[:, i, :]
        ref_traj: np.ndarray = y_ref[:, i, :]
        abs_err: np.ndarray = np.abs(zig_traj - ref_traj)
        max_abs: float = float(abs_err.max())
        denom: np.ndarray = np.maximum(np.abs(ref_traj), 1e-15)
        max_rel: float = float((abs_err / denom).max())
        overall_max_abs = max(overall_max_abs, max_abs)
        print(
            f"  {str(ic):>20} {max_abs:>12.4e} {max_rel:>12.4e} "
            f"{zig_traj[-1, 0]:>14.6f} {ref_traj[-1, 0]:>14.6f}"
        )

    # Two independent Dopri5 implementations may diverge in phase for oscillatory
    # trajectories over long integration spans due to different step-size controllers.
    # Observed max ≈ 0.046 between Zig and JAX; 0.5 gives ~10× margin.
    report(
        f"[{label}] all close (atol={CROSS_SOLVER_ATOL}, long-span phase drift)",
        np.allclose(y_zig_t, y_ref, atol=CROSS_SOLVER_ATOL),
        f"max_abs={overall_max_abs:.4e}",
    )

    # Fixed-point ICs converge to a stable equilibrium; solvers must agree
    # to near-solver-tolerance precision. Observed max ≈ 2e-5; 1e-3 gives margin.
    fp_indices: list[int] = [0, 4]
    fp_max: float = 0.0
    for i in fp_indices:
        abs_err_i: float = float(np.abs(y_zig_t[:, i, :] - y_ref[:, i, :]).max())
        fp_max = max(fp_max, abs_err_i)
    report(
        f"[{label}] fixed-point ICs close (atol={FIXED_POINT_ATOL})",
        fp_max < FIXED_POINT_ATOL,
        f"max_abs={fp_max:.4e}",
    )

    lc_indices: list[int] = [1, 2, 3]
    omega_match = True
    for i in lc_indices:
        zig_omega: float = float(y_zig_t[-1, i, 1])
        ref_omega: float = float(y_ref[-1, i, 1])
        if not np.isclose(zig_omega, ref_omega, atol=LIMIT_CYCLE_OMEGA_ATOL):
            omega_match = False
    report(
        f"[{label}] limit-cycle ICs same attractor (ω atol={LIMIT_CYCLE_OMEGA_ATOL})", omega_match
    )


def test_numerical_correctness() -> None:
    print("\n--- Numerical correctness: Zig vs JAX & torchdiffeq ---")

    y0s = np.array(TEST_ICS, dtype=np.float64)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)

    t_zig, y_zig = solve_with_zig(y0s, t_eval)
    y_zig_t: np.ndarray = y_zig.transpose(1, 0, 2)

    # --- vs JAX/Diffrax ---
    print("\n  >> Zig vs JAX/Diffrax (Dopri5 + PID controller)")
    t_jax, y_jax = solve_with_jax(y0s)
    _compare_solver_pair("JAX", y_zig_t, y_jax, t_zig, t_jax)

    # --- vs torchdiffeq ---
    print("\n  >> Zig vs torchdiffeq (Dopri5 + classic controller)")
    t_tde, y_tde = solve_with_torchdiffeq(y0s)
    _compare_solver_pair("torchdiffeq", y_zig_t, y_tde, t_zig, t_tde)

    # --- JAX vs torchdiffeq (triangulation) ---
    print("\n  >> JAX vs torchdiffeq (cross-check, no Zig)")
    _compare_solver_pair("JAX↔torchdiffeq", y_jax, y_tde, t_jax, t_tde)

    # --- Early-time divergence check ---
    # At short integration times all solvers should agree to near-solver-
    # tolerance precision (observed max ≈ 2.5e-5; 1e-3 gives ample margin).
    print("\n  >> Early-time check (t=10, step index ~10)")
    early_idx = 10
    early_max: float = 0.0
    print(f"  {'IC':>20} {'Zig-JAX @ t=10':>16} {'Zig-TDE @ t=10':>16} {'JAX-TDE @ t=10':>16}")
    print("  " + "-" * 72)
    for i, ic in enumerate(TEST_ICS):
        zig_jax = float(np.abs(y_zig_t[early_idx, i, :] - y_jax[early_idx, i, :]).max())
        zig_tde = float(np.abs(y_zig_t[early_idx, i, :] - y_tde[early_idx, i, :]).max())
        jax_tde = float(np.abs(y_jax[early_idx, i, :] - y_tde[early_idx, i, :]).max())
        early_max = max(early_max, zig_jax, zig_tde, jax_tde)
        print(f"  {str(ic):>20} {zig_jax:>16.6e} {zig_tde:>16.6e} {jax_tde:>16.6e}")
    report(
        f"early-time all pairs close (atol={EARLY_TIME_ATOL})",
        early_max < EARLY_TIME_ATOL,
        f"max={early_max:.4e}",
    )


def test_determinism() -> None:
    print("\n--- Determinism ---")
    solver = ZigODESolver(auto_rebuild_solver=False)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)
    y0s = np.array(TEST_ICS, dtype=np.float64)

    _, y1 = solver.solve(PENDULUM_ZIG, y0s, T_SPAN, t_eval, params=PARAMS)
    _, y2 = solver.solve(PENDULUM_ZIG, y0s, T_SPAN, t_eval, params=PARAMS)
    report("two runs identical", np.array_equal(y1, y2))


def test_performance() -> None:
    print("\n--- Performance (single IC, 1000 steps) ---")
    solver = ZigODESolver(auto_rebuild_solver=False)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)

    # Warm-up
    solver.solve(PENDULUM_ZIG, [0.4, 0.0], T_SPAN, t_eval, params=PARAMS)

    n_runs = 500
    start = time.perf_counter()
    for _ in range(n_runs):
        solver.solve(PENDULUM_ZIG, [0.4, 0.0], T_SPAN, t_eval, params=PARAMS)
    elapsed = time.perf_counter() - start
    per_call_us = elapsed / n_runs * 1e6
    report(f"{n_runs} single-IC solves", True, f"{per_call_us:.0f} µs/call")


# ============================================================
# SymPy ODE tests
# ============================================================


def test_sympy_codegen() -> None:
    """Test that SymPyODE generates valid C source code."""
    print("\n--- SymPy ODE codegen tests ---")

    c_src: str = PENDULUM_SYMPY.to_c_source()
    report("C source contains ode_func", "ode_func" in c_src)
    report("C source contains ode_dim", "ode_dim" in c_src)
    report("C source contains ode_param_size", "ode_param_size" in c_src)
    report("C source contains sin(", "sin(" in c_src)
    report("source_exists before write", PENDULUM_SYMPY.source_exists())
    report("param_names correct", PENDULUM_SYMPY.param_names == ["alpha", "T", "K"])


def test_sympy_shapes() -> None:
    """Test that a SymPy-compiled ODE gives correct output shapes."""
    print("\n--- SymPy ODE shape tests ---")

    solver = ZigODESolver(auto_rebuild_solver=False)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)

    # Single IC
    _, y = solver.solve(PENDULUM_SYMPY, [0.4, 0.0], T_SPAN, t_eval, params=PARAMS)
    report("sympy single IC 3-D", y.ndim == 3, f"shape={y.shape}")
    report("sympy single IC shape (1, N, 2)", y.shape == (1, N_STEPS, 2), f"shape={y.shape}")

    # Batch IC
    y0s = np.array(TEST_ICS, dtype=np.float64)
    _, y = solver.solve(PENDULUM_SYMPY, y0s, T_SPAN, t_eval, params=PARAMS)
    report(
        "sympy batch shape (5, N, 2)",
        y.shape == (len(TEST_ICS), N_STEPS, 2),
        f"shape={y.shape}",
    )


def test_sympy_vs_zig() -> None:
    """Test that the SymPy-generated C ODE produces results very close to the Zig ODE.

    Both go through the same Dopri5 solver with the same tolerances. The RHS
    expressions are mathematically identical, but the compilers (``cc`` vs ``zig``)
    may produce slightly different ``sin()`` implementations, leading to tiny
    floating-point differences that accumulate over long integration spans.
    """
    print("\n--- SymPy ODE vs Zig ODE ---")

    solver = ZigODESolver(auto_rebuild_solver=False)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)
    y0s = np.array(TEST_ICS, dtype=np.float64)

    _, y_zig = solver.solve(PENDULUM_ZIG, y0s, T_SPAN, t_eval, params=PARAMS, rtol=RTOL, atol=ATOL)
    _, y_sympy = solver.solve(
        PENDULUM_SYMPY, y0s, T_SPAN, t_eval, params=PARAMS, rtol=RTOL, atol=ATOL
    )

    max_abs: float = float(np.abs(y_zig - y_sympy).max())
    report(
        f"Zig vs SymPy close (atol={SYMPY_VS_ZIG_ATOL})",
        max_abs < SYMPY_VS_ZIG_ATOL,
        f"max_abs={max_abs:.4e}",
    )

    for i, ic in enumerate(TEST_ICS):
        diff_i: float = float(np.abs(y_zig[i] - y_sympy[i]).max())
        report(f"  IC {ic} max|err|={diff_i:.4e}", diff_i < SYMPY_VS_ZIG_ATOL)


def test_sympy_determinism() -> None:
    """Test that repeated SymPy ODE solves give identical results."""
    print("\n--- SymPy ODE determinism ---")

    solver = ZigODESolver(auto_rebuild_solver=False)
    t_eval = np.linspace(T_SPAN[0], T_SPAN[1], N_STEPS)
    y0s = np.array(TEST_ICS, dtype=np.float64)

    _, y1 = solver.solve(PENDULUM_SYMPY, y0s, T_SPAN, t_eval, params=PARAMS)
    _, y2 = solver.solve(PENDULUM_SYMPY, y0s, T_SPAN, t_eval, params=PARAMS)
    report("sympy two runs identical", np.array_equal(y1, y2))


def main() -> None:
    print("=" * 60)
    print("Zig ODE Solver — Test Suite")
    print("=" * 60)

    test_shapes()
    test_numerical_correctness()
    test_determinism()
    test_performance()

    test_sympy_codegen()
    test_sympy_shapes()
    test_sympy_vs_zig()
    test_sympy_determinism()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
