# pyright: basic
"""
Experiment: torchdiffeq vs torchode for pendulum basin stability.

Compares plain-function ODE vs nn.Module-wrapped ODE for both solvers.
Configs: CPU and GPU (if available) × {10k, 20k} initial conditions × 3 runs.

Pendulum ODE:
    dθ/dt   = θ̇
    dθ̇/dt  = -α·θ̇ + T - K·sin(θ)

Parameters match setup_pendulum_system.py: alpha=0.1, T=0.5, K=1.0.
Integration: t_span=(0, 1000), save region=(950, 1000) with 50 points, dopri5.
Tolerances: rtol=1e-8, atol=1e-6.
"""

import time

import torch
import torchode as to  # type: ignore[import-untyped]
from torchdiffeq import odeint  # type: ignore[import-untyped]

from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = 0.1
TORQUE = 0.5
K = 1.0

T_START = 0.0
T_END = 1000.0
T_EVAL_START = 950.0
T_EVAL_END = 1000.0
T_EVAL_STEPS = 50

RTOL = 1e-8
ATOL = 1e-6
N_RUNS = 1

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# ODE — plain function, no nn.Module
# ---------------------------------------------------------------------------
def pendulum_ode(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pendulum RHS expecting y of shape (batch, 2)."""
    theta = y[:, 0]
    theta_dot = y[:, 1]
    dtheta_dot = -ALPHA * theta_dot + TORQUE - K * torch.sin(theta)
    return torch.stack([theta_dot, dtheta_dot], dim=1)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def make_y0(n: int, device: str) -> torch.Tensor:
    """Uniform grid of n initial conditions across the state space."""
    theta = torch.linspace(-torch.pi + 0.5, torch.pi + 0.5, n, dtype=DTYPE, device=device)
    theta_dot = torch.zeros(n, dtype=DTYPE, device=device)
    return torch.stack([theta, theta_dot], dim=1)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def solve_torchdiffeq_fn(y0: torch.Tensor, device: str) -> torch.Tensor:
    save_ts = torch.linspace(T_EVAL_START, T_EVAL_END, T_EVAL_STEPS, dtype=DTYPE, device=device)
    anchor = torch.tensor([T_START], dtype=DTYPE, device=device)
    t_eval = torch.cat([anchor, save_ts])

    with torch.no_grad():
        y: torch.Tensor = odeint(  # type: ignore[assignment]
            pendulum_ode, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL
        )
    return y[1:]  # drop anchor — shape (T_EVAL_STEPS, batch, 2)


def solve_torchdiffeq_nn(y0: torch.Tensor, device: str) -> torch.Tensor:
    params: PendulumParams = {"alpha": ALPHA, "T": TORQUE, "K": K}
    module = PendulumODE(params).to(device)
    save_ts = torch.linspace(T_EVAL_START, T_EVAL_END, T_EVAL_STEPS, dtype=DTYPE, device=device)
    anchor = torch.tensor([T_START], dtype=DTYPE, device=device)
    t_eval = torch.cat([anchor, save_ts])

    with torch.no_grad():
        y: torch.Tensor = odeint(  # type: ignore[assignment]
            module, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL
        )
    return y[1:]


def solve_torchode_fn(y0: torch.Tensor, device: str) -> torch.Tensor:
    batch_size = y0.shape[0]
    save_ts = torch.linspace(T_EVAL_START, T_EVAL_END, T_EVAL_STEPS, dtype=DTYPE, device=device)

    t_start = torch.full((batch_size,), T_START, dtype=DTYPE, device=device)
    t_end = torch.full((batch_size,), T_EVAL_END, dtype=DTYPE, device=device)
    save_ts_batched = save_ts.unsqueeze(0).expand(batch_size, -1)

    term = to.ODETerm(pendulum_ode)  # pyright: ignore[reportArgumentType]
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=ATOL, rtol=RTOL, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # pyright: ignore[reportArgumentType]

    problem = to.InitialValueProblem(
        y0=y0,  # pyright: ignore[reportArgumentType]
        t_start=t_start,  # pyright: ignore[reportArgumentType]
        t_end=t_end,  # pyright: ignore[reportArgumentType]
        t_eval=save_ts_batched,  # pyright: ignore[reportArgumentType]
    )

    with torch.inference_mode():
        solution = solver.solve(problem)
        return solution.ys.transpose(0, 1)  # (T_EVAL_STEPS, batch, 2)


def solve_torchode_nn(y0: torch.Tensor, device: str) -> torch.Tensor:
    params: PendulumParams = {"alpha": ALPHA, "T": TORQUE, "K": K}
    module = PendulumODE(params).to(device)
    batch_size = y0.shape[0]
    save_ts = torch.linspace(T_EVAL_START, T_EVAL_END, T_EVAL_STEPS, dtype=DTYPE, device=device)

    t_start = torch.full((batch_size,), T_START, dtype=DTYPE, device=device)
    t_end = torch.full((batch_size,), T_EVAL_END, dtype=DTYPE, device=device)
    save_ts_batched = save_ts.unsqueeze(0).expand(batch_size, -1)

    term = to.ODETerm(module)  # pyright: ignore[reportArgumentType]
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=ATOL, rtol=RTOL, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # pyright: ignore[reportArgumentType]

    problem = to.InitialValueProblem(
        y0=y0,  # pyright: ignore[reportArgumentType]
        t_start=t_start,  # pyright: ignore[reportArgumentType]
        t_end=t_end,  # pyright: ignore[reportArgumentType]
        t_eval=save_ts_batched,  # pyright: ignore[reportArgumentType]
    )

    with torch.inference_mode():
        solution = solver.solve(problem)
        return solution.ys.transpose(0, 1)


# ---------------------------------------------------------------------------
# Solvers — torch.compile variants
# Each returns (compile_run_time, post_compile_time): the ODE fn is compiled
# once per call to run_compiled_benchmark, so call 1 = compile+run, call 2 = run.
# ---------------------------------------------------------------------------
def _odeint_with(ode_fn: object, y0: torch.Tensor, device: str) -> torch.Tensor:
    """Run torchdiffeq odeint with any callable ODE function."""
    save_ts = torch.linspace(T_EVAL_START, T_EVAL_END, T_EVAL_STEPS, dtype=DTYPE, device=device)
    anchor = torch.tensor([T_START], dtype=DTYPE, device=device)
    t_eval = torch.cat([anchor, save_ts])
    with torch.no_grad():
        y: torch.Tensor = odeint(ode_fn, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL)  # type: ignore[assignment]
    return y[1:]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
SOLVERS: dict[str, object] = {
    "torchdiffeq/fn": solve_torchdiffeq_fn,
    "torchdiffeq/nn": solve_torchdiffeq_nn,
    "torchode/fn": solve_torchode_fn,
    "torchode/nn": solve_torchode_nn,
}


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_benchmark(solver_name: str, n: int, device: str) -> list[float]:
    from collections.abc import Callable

    y0 = make_y0(n, device)
    fn: Callable[[torch.Tensor, str], torch.Tensor] = SOLVERS[solver_name]  # type: ignore[assignment]

    times: list[float] = []
    for _ in range(N_RUNS):
        _sync(device)
        t0 = time.perf_counter()
        _ = fn(y0, device)
        _sync(device)
        times.append(time.perf_counter() - t0)

    return times


def run_compiled_benchmark(solver_name: str, n: int, device: str) -> tuple[float, float]:
    """Compile once, call twice: returns (compile+run time, post-compile time)."""
    y0 = make_y0(n, device)

    if solver_name == "torchdiffeq/fn+compile":
        compiled_ode: object = torch.compile(pendulum_ode)
    elif solver_name == "torchdiffeq/nn+compile":
        params: PendulumParams = {"alpha": ALPHA, "T": TORQUE, "K": K}
        compiled_ode = torch.compile(PendulumODE(params).to(device))
    else:
        raise ValueError(f"Unknown compiled solver: {solver_name}")

    # Call 1: triggers JIT compilation
    _sync(device)
    t0 = time.perf_counter()
    _ = _odeint_with(compiled_ode, y0, device)
    _sync(device)
    t_compile = time.perf_counter() - t0

    # Call 2: compiled path (cache hit)
    _sync(device)
    t0 = time.perf_counter()
    _ = _odeint_with(compiled_ode, y0, device)
    _sync(device)
    t_compiled = time.perf_counter() - t0

    return t_compile, t_compiled


SOLVER_ORDER: list[str] = [
    "torchdiffeq/fn",
    "torchdiffeq/nn",
    "torchode/fn",
    "torchode/nn",
]

COMPILED_SOLVERS: list[str] = ["torchdiffeq/fn+compile", "torchdiffeq/nn+compile"]


def main() -> None:
    devices: list[str] = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("CUDA not available — running CPU only.\n")

    n_points_list: list[int] = [10_000, 20_000]

    run_headers = "  ".join(f"{'Run ' + str(i + 1):>8}" for i in range(N_RUNS))
    header = f"{'Solver':<24} {'Device':<8} {'N':>8}  {run_headers}  {'Mean':>8}"
    compiled_header = f"{'Solver':<24} {'Device':<8} {'N':>8}  {'1st(+JIT)':>10}  {'2nd':>8}"
    sep = "-" * (46 + 10 * N_RUNS)
    compiled_sep = "-" * 60

    for device in devices:
        print(f"\n=== Device: {device} ===")
        print()
        for n in n_points_list:
            print(f"--- N = {n:,} ---")
            print(header)
            print(sep)
            for solver_name in SOLVER_ORDER:
                if solver_name in COMPILED_SOLVERS:
                    continue
                times = run_benchmark(solver_name, n, device)
                mean_t = sum(times) / len(times)
                run_cols = "  ".join(f"{t:>6.3f}s" for t in times)
                print(f"{solver_name:<24} {device:<8} {n:>8,}  {run_cols}  {mean_t:>6.3f}s")
            print()
            print(compiled_header)
            print(compiled_sep)
            for solver_name in SOLVER_ORDER:
                if solver_name not in COMPILED_SOLVERS:
                    continue
                t_compile, t_compiled = run_compiled_benchmark(solver_name, n, device)
                print(
                    f"{solver_name:<24} {device:<8} {n:>8,}  {t_compile:>8.3f}s  {t_compiled:>6.3f}s"
                )
            print()


if __name__ == "__main__":
    main()
