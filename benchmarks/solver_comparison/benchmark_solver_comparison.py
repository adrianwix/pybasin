# pyright: basic
# pyright: reportReturnType=false
# pyright: reportArgumentType=false
"""
Pytest-benchmark comparing different ODE solver libraries for pendulum integration.

Benchmarks ONLY the raw solver libraries (no pybasin abstractions):
- JAX/Diffrax (Dopri5) - CPU and CUDA
- torchdiffeq (dopri5) - CPU and CUDA
- torchode (dopri5) - CUDA only (no CPU support)

Hardware:
    CPU: Intel(R) Core(TM) Ultra 9 275HX
    GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU

Run with:
    uv run pytest benchmarks/solver_comparison/benchmark_solver_comparison.py --benchmark-only
    uv run pytest benchmarks/solver_comparison/benchmark_solver_comparison.py --benchmark-only --benchmark-json=benchmarks/solver_comparison/results/benchmark_results.json

Compare results:
    uv run pytest-benchmark compare
"""

# Benchmark settings (5 rounds)
BENCHMARK_ROUNDS = 5
BENCHMARK_WARMUP_ROUNDS = 0


import logging
from typing import Any

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_compilation_cache", False)
import numpy as np
import pytest
import torch
import torchode as to
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from torchdiffeq import odeint

logging.getLogger("pybasin").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)

# Benchmark configuration
N_VALUES = [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_VALUES_TORCHODE = [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]

# ODE parameters (same for all solvers)
ALPHA = 0.1
T = 0.5
K = 1.0

# Integration settings
TIME_SPAN = (0.0, 1000.0)
N_STEPS = 1000
RTOL = 1e-8
ATOL = 1e-6

# Sampling region
MIN_LIMITS = [-np.pi + np.arcsin(T / K), -10.0]
MAX_LIMITS = [np.pi + np.arcsin(T / K), 10.0]


# --- ODE definitions for each library ---


def ode_jax(t: jax.Array, y: jax.Array, args: None) -> jax.Array:
    """JAX/Diffrax pendulum ODE."""
    theta, omega = y[0], y[1]
    dtheta = omega
    domega = -ALPHA * omega + T - K * jnp.sin(theta)
    return jnp.array([dtheta, domega])


def ode_torch(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch pendulum ODE - batched (used by torchdiffeq and torchode)."""
    theta = y[..., 0]
    omega = y[..., 1]
    dtheta = omega
    domega = -ALPHA * omega + T - K * torch.sin(theta)
    return torch.stack([dtheta, domega], dim=-1)


# --- Initial conditions generation ---


def generate_y0_numpy(n: int) -> np.ndarray:
    """Generate initial conditions as numpy array."""
    y0 = np.empty((n, 2), dtype=np.float64)
    y0[:, 0] = np.random.uniform(MIN_LIMITS[0], MAX_LIMITS[0], n)
    y0[:, 1] = np.random.uniform(MIN_LIMITS[1], MAX_LIMITS[1], n)
    return y0


def generate_y0_torch(n: int, device: str) -> torch.Tensor:
    """Generate initial conditions as torch tensor."""
    dtype = torch.float32 if device == "cuda" else torch.float64
    y0 = torch.empty((n, 2), dtype=dtype, device=device)
    y0[:, 0].uniform_(MIN_LIMITS[0], MAX_LIMITS[0])
    y0[:, 1].uniform_(MIN_LIMITS[1], MAX_LIMITS[1])
    return y0


# --- JAX/Diffrax (Dopri5) ---


def run_jax_diffrax(y0_jax: jax.Array, device: str) -> None:
    jax_device = jax.devices("gpu")[0] if device == "cuda" else jax.devices("cpu")[0]

    if device == "cuda":
        mem = jax_device.memory_stats()
        if mem:
            alloc_gb = mem["bytes_in_use"] / 1e9
            peak_gb = mem["peak_bytes_in_use"] / 1e9
            print(f"  [jax] BEFORE solve: in_use={alloc_gb:.2f} GB, peak={peak_gb:.2f} GB")

    dtype = jnp.float32 if device == "cuda" else jnp.float64
    t_eval = jnp.linspace(TIME_SPAN[0], TIME_SPAN[1], N_STEPS, dtype=dtype)
    t_eval = jax.device_put(t_eval, jax_device)

    term = ODETerm(ode_jax)
    solver = Dopri5()
    stepsize_controller = PIDController(rtol=RTOL, atol=ATOL)
    saveat = SaveAt(ts=t_eval)

    def solve_single(y0_single: jax.Array) -> jax.Array:
        sol = diffeqsolve(
            term,
            solver,
            t0=TIME_SPAN[0],
            t1=TIME_SPAN[1],
            dt0=0.1,
            y0=y0_single,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=16**5,
        )
        return sol.ys

    solve_batch = jax.vmap(solve_single)
    y_result = solve_batch(y0_jax)
    y_result.block_until_ready()

    if device == "cuda":
        mem = jax_device.memory_stats()
        if mem:
            alloc_gb = mem["bytes_in_use"] / 1e9
            peak_gb = mem["peak_bytes_in_use"] / 1e9
            print(f"  [jax] AFTER solve:  in_use={alloc_gb:.2f} GB, peak={peak_gb:.2f} GB")

    del y_result, t_eval


# --- torchdiffeq (dopri5) ---


def run_torchdiffeq(y0: torch.Tensor, device: str) -> None:
    dtype = torch.float32 if device == "cuda" else torch.float64
    t_eval = torch.linspace(TIME_SPAN[0], TIME_SPAN[1], N_STEPS, device=device, dtype=dtype)

    if device == "cuda":
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(
            f"  [torchdiffeq] BEFORE solve: allocated={alloc_gb:.2f} GB, reserved={reserved_gb:.2f} GB"
        )

    with torch.no_grad():
        y_result = odeint(ode_torch, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL)

    if device == "cuda":
        torch.cuda.synchronize()
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(
            f"  [torchdiffeq] AFTER solve:  allocated={alloc_gb:.2f} GB, reserved={reserved_gb:.2f} GB"
        )

    del y_result, t_eval
    if device == "cuda":
        torch.cuda.empty_cache()


# --- torchode (dopri5) ---


def run_torchode(y0: torch.Tensor, device: str) -> None:
    n = y0.shape[0]
    dtype = torch.float32 if device == "cuda" else torch.float64
    t_eval = torch.linspace(TIME_SPAN[0], TIME_SPAN[1], N_STEPS, device=device, dtype=dtype)

    if device == "cuda":
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(
            f"  [torchode] BEFORE solve: allocated={alloc_gb:.2f} GB, reserved={reserved_gb:.2f} GB"
        )

    with torch.inference_mode():
        term = to.ODETerm(ode_torch)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(atol=ATOL, rtol=RTOL, term=term)
        adjoint = to.AutoDiffAdjoint(step_method, step_size_controller)

        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval.repeat((n, 1)))
        sol = adjoint.solve(problem)

    if device == "cuda":
        torch.cuda.synchronize()
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(
            f"  [torchode] AFTER solve:  allocated={alloc_gb:.2f} GB, reserved={reserved_gb:.2f} GB"
        )

    del sol, term, step_method, step_size_controller, adjoint, problem, t_eval
    if device == "cuda":
        torch.cuda.empty_cache()


# --- Pytest Benchmark Tests ---


def make_counter_wrapper(func: Any, total_rounds: int) -> tuple[Any, dict[str, int]]:
    """
    Create a wrapper function that prints round numbers during benchmarking.

    :param func: The function to wrap.
    :param total_rounds: Total number of rounds that will be executed.
    :return: Tuple of (wrapped function, counter dict).
    """
    counter = {"count": 0}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        counter["count"] += 1
        print(f"  Round {counter['count']}/{total_rounds}")
        return func(*args, **kwargs)

    return wrapper, counter


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_benchmark_jax_diffrax(benchmark: Any, n: int, device: str) -> None:
    if device == "cuda" and not jax.devices("gpu"):
        pytest.skip("CUDA not available for JAX")

    print(f"\nRunning: JAX/Diffrax, n={n}, device={device}")
    jax_device = jax.devices("gpu")[0] if device == "cuda" else jax.devices("cpu")[0]
    y0_np = generate_y0_numpy(n)
    dtype = jnp.float32 if device == "cuda" else jnp.float64
    y0_jax = jax.device_put(jnp.array(y0_np, dtype=dtype), jax_device)

    run_with_counter, _ = make_counter_wrapper(run_jax_diffrax, BENCHMARK_ROUNDS)

    benchmark.pedantic(
        run_with_counter,
        args=(y0_jax, device),
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=0,
    )


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_benchmark_torchdiffeq(benchmark: Any, n: int, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    print(f"\nRunning: torchdiffeq, n={n}, device={device}")
    y0 = generate_y0_torch(n, device)

    run_with_counter, _ = make_counter_wrapper(run_torchdiffeq, BENCHMARK_ROUNDS)

    benchmark.pedantic(
        run_with_counter,
        args=(y0, device),
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=0,
    )


"""
The first run for 100k after all the previous runs was slow. For sure there is an issue here:
"data": [
    582.87340632,
    309.84245656999974,
    310.4242675710002,
    310.13977866799996,
    309.4117136310001
],
"""


@pytest.mark.parametrize("n", N_VALUES_TORCHODE)
def test_benchmark_torchode_cuda(benchmark: Any, n: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    print(f"\nRunning: torchode, n={n}, device=cuda")
    y0 = generate_y0_torch(n, "cuda")

    run_with_counter, _ = make_counter_wrapper(run_torchode, BENCHMARK_ROUNDS)

    benchmark.pedantic(
        run_with_counter,
        args=(y0, "cuda"),
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=0,
    )


@pytest.mark.skip(reason="CPU benchmark skipped")
def test_benchmark_torchode_cpu(benchmark: Any) -> None:
    print("\nRunning: torchode, n=10000, device=cpu")
    n = 10_000
    y0 = generate_y0_torch(n, "cpu")

    run_with_counter, _ = make_counter_wrapper(run_torchode, BENCHMARK_ROUNDS)

    benchmark.pedantic(
        run_with_counter,
        args=(y0, "cpu"),
        rounds=BENCHMARK_ROUNDS,
        warmup_rounds=0,
    )
    result = run_torchode(y0, "cpu")
    assert result is not None
    assert result.shape[1] == N_STEPS
