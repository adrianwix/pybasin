# pyright: basic
# pyright: reportMissingModuleSource=false
# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false
"""
Experiment: Parameter batching for ODE integration.

Tests batching over parameter grids with a single initial condition using:
1. Diffrax (JAX) with vmap
2. torchdiffeq (PyTorch) with vectorized ODE
3. torchode (PyTorch) with per-sample adaptive stepping
4. SciPy solve_ivp (CPU, sequential)

Uses the pendulum ODE as the test case:
    dθ/dt = θ̇
    dθ̇/dt = -α·θ̇ + T - K·sin(θ)
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchode as to  # type: ignore[import-untyped]
from diffrax import (  # type: ignore[import-untyped]
    Dopri5,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]
from torchdiffeq import odeint  # type: ignore[import-untyped]

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])


def run_diffrax_parameter_batching() -> None:
    """Test parameter batching with Diffrax using vmap."""
    print("=" * 60)
    print("Diffrax Parameter Batching (JAX + vmap)")
    print("=" * 60)

    def pendulum_ode(
        t: Any, y: jax.Array, args: tuple[jax.Array, jax.Array, jax.Array]
    ) -> jax.Array:
        alpha, torque, k = args
        theta, theta_dot = y[0], y[1]
        dtheta = theta_dot
        dtheta_dot = -alpha * theta_dot + torque - k * jnp.sin(theta)
        return jnp.array([dtheta, dtheta_dot])

    def solve_single(params: jax.Array) -> jax.Array:
        alpha, torque, k = params[0], params[1], params[2]
        term = ODETerm(lambda t, y, _: pendulum_ode(t, y, (alpha, torque, k)))
        saveat = SaveAt(ts=jnp.linspace(0, 10, 100))
        sol = diffeqsolve(
            term,
            Tsit5(),
            t0=0.0,
            t1=10.0,
            dt0=0.01,
            y0=jnp.array([0.1, 0.0]),
            saveat=saveat,
        )
        assert isinstance(sol.ys, jax.Array)
        return sol.ys

    alphas = jnp.linspace(0.1, 1.0, 5)
    torques = jnp.linspace(0.0, 2.0, 5)
    k_fixed = 1.0

    alpha_grid, torque_grid = jnp.meshgrid(alphas, torques, indexing="ij")
    param_grid = jnp.stack(
        [alpha_grid.flatten(), torque_grid.flatten(), jnp.full(25, k_fixed)],
        axis=1,
    )

    print(f"Parameter grid shape: {param_grid.shape}")
    print(f"  - alphas: {alphas}")
    print(f"  - torques: {torques}")
    print(f"  - K (fixed): {k_fixed}")

    batched_solve = jax.vmap(solve_single)

    print("\nCompiling (first run)...")
    start = time.perf_counter()
    trajectories = batched_solve(param_grid)
    trajectories.block_until_ready()
    compile_time = time.perf_counter() - start
    print(f"Compile + first run time: {compile_time:.3f}s")

    print("\nRunning batched solve (compiled)...")
    start = time.perf_counter()
    trajectories = batched_solve(param_grid)
    trajectories.block_until_ready()
    run_time = time.perf_counter() - start

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}")
    print("  - (n_param_sets, n_timesteps, state_dim)")

    print("\nSample final states (first 5 parameter sets):")
    for i in range(5):
        final_state = trajectories[i, -1, :]
        print(
            f"  params[{i}] (α={param_grid[i, 0]:.2f}, T={param_grid[i, 1]:.2f}): "
            f"θ={final_state[0]:.4f}, θ̇={final_state[1]:.4f}"
        )


class BatchedPendulumTorch(torch.nn.Module):
    """Vectorized pendulum ODE for torchdiffeq parameter batching."""

    def __init__(self, alphas: torch.Tensor, torques: torch.Tensor, ks: torch.Tensor):
        super().__init__()
        self.alphas = alphas
        self.torques = torques
        self.ks = ks

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        theta = y[:, 0]
        theta_dot = y[:, 1]

        dtheta = theta_dot
        dtheta_dot = -self.alphas * theta_dot + self.torques - self.ks * torch.sin(theta)

        return torch.stack([dtheta, dtheta_dot], dim=1)


def run_torchdiffeq_parameter_batching() -> None:
    """Test parameter batching with torchdiffeq using vectorized ODE."""
    print("\n" + "=" * 60)
    print("torchdiffeq Parameter Batching (PyTorch)")
    print("=" * 60)

    alphas = torch.linspace(0.1, 1.0, 5, dtype=torch.float64)
    torques = torch.linspace(0.0, 2.0, 5, dtype=torch.float64)
    k_fixed = 1.0

    alpha_grid, torque_grid = torch.meshgrid(alphas, torques, indexing="ij")
    alpha_flat = alpha_grid.flatten()
    torque_flat = torque_grid.flatten()
    k_flat = torch.full((25,), k_fixed, dtype=torch.float64)

    n_params = alpha_flat.shape[0]

    print(f"Parameter grid shape: ({n_params}, 3)")
    print(f"  - alphas: {alphas.tolist()}")
    print(f"  - torques: {torques.tolist()}")
    print(f"  - K (fixed): {k_fixed}")

    y0_single = torch.tensor([0.1, 0.0], dtype=torch.float64)
    y0 = y0_single.unsqueeze(0).expand(n_params, -1).clone()

    print(f"Initial condition (broadcast): {y0_single.tolist()} -> shape {y0.shape}")

    ode = BatchedPendulumTorch(alpha_flat, torque_flat, k_flat)
    t_span = torch.linspace(0, 10, 100, dtype=torch.float64)

    print("\nRunning batched solve...")
    start = time.perf_counter()
    trajectories_result = odeint(ode, y0, t_span, method="dopri5")
    assert isinstance(trajectories_result, torch.Tensor)
    trajectories = trajectories_result
    run_time = time.perf_counter() - start

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}")
    print("  - (n_timesteps, n_param_sets, state_dim)")

    trajectories_reordered = trajectories.permute(1, 0, 2)
    print(f"Reordered shape: {trajectories_reordered.shape}")
    print("  - (n_param_sets, n_timesteps, state_dim)")

    print("\nSample final states (first 5 parameter sets):")
    for i in range(5):
        final_state = trajectories_reordered[i, -1, :]
        print(
            f"  params[{i}] (α={alpha_flat[i]:.2f}, T={torque_flat[i]:.2f}): "
            f"θ={final_state[0]:.4f}, θ̇={final_state[1]:.4f}"
        )


def run_torchode_parameter_batching() -> None:
    """Test parameter batching with torchode using per-sample adaptive stepping."""
    print("\n" + "=" * 60)
    print("torchode Parameter Batching (PyTorch, per-sample steps)")
    print("=" * 60)

    alphas = torch.linspace(0.1, 1.0, 5, dtype=torch.float64)
    torques = torch.linspace(0.0, 2.0, 5, dtype=torch.float64)
    k_fixed = 1.0

    alpha_grid, torque_grid = torch.meshgrid(alphas, torques, indexing="ij")
    alpha_flat = alpha_grid.flatten()
    torque_flat = torque_grid.flatten()
    k_flat = torch.full((25,), k_fixed, dtype=torch.float64)

    n_params = alpha_flat.shape[0]

    print(f"Parameter grid shape: ({n_params}, 3)")
    print(f"  - alphas: {alphas.tolist()}")
    print(f"  - torques: {torques.tolist()}")
    print(f"  - K (fixed): {k_fixed}")

    y0_single = torch.tensor([0.1, 0.0], dtype=torch.float64)
    y0 = y0_single.unsqueeze(0).expand(n_params, -1).clone()

    print(f"Initial condition (broadcast): {y0_single.tolist()} -> shape {y0.shape}")

    ode = BatchedPendulumTorch(alpha_flat, torque_flat, k_flat)
    n_steps = 100
    t_eval = torch.linspace(0, 10, n_steps, dtype=torch.float64)
    t_eval_batched = t_eval.unsqueeze(0).expand(n_params, -1)

    term = to.ODETerm(ode)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-8, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # pyright: ignore[reportArgumentType]

    problem = to.InitialValueProblem(
        y0=y0,  # pyright: ignore[reportArgumentType]
        t_start=torch.full((n_params,), 0.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
        t_end=torch.full((n_params,), 10.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
        t_eval=t_eval_batched,  # pyright: ignore[reportArgumentType]
    )

    print("\nRunning batched solve...")
    start = time.perf_counter()
    solution = solver.solve(problem)
    run_time = time.perf_counter() - start

    trajectories = solution.ys  # (batch, n_steps, n_dims)

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}")
    print("  - (n_param_sets, n_timesteps, state_dim)")

    print("\nSample final states (first 5 parameter sets):")
    for i in range(5):
        final_state = trajectories[i, -1, :]
        print(
            f"  params[{i}] (\u03b1={alpha_flat[i]:.2f}, T={torque_flat[i]:.2f}): "
            f"\u03b8={final_state[0]:.4f}, \u03b8\u0307={final_state[1]:.4f}"
        )


def run_scipy_parameter_batching() -> None:
    """Test parameter batching with SciPy solve_ivp (sequential, CPU-only)."""
    print("\n" + "=" * 60)
    print("SciPy Parameter Batching (CPU, sequential solve_ivp)")
    print("=" * 60)

    alphas = np.linspace(0.1, 1.0, 5)
    torques_arr = np.linspace(0.0, 2.0, 5)
    k_fixed = 1.0

    alpha_grid, torque_grid = np.meshgrid(alphas, torques_arr, indexing="ij")
    alpha_flat = alpha_grid.flatten()
    torque_flat = torque_grid.flatten()

    n_params = alpha_flat.shape[0]

    print(f"Parameter grid shape: ({n_params}, 3)")
    print(f"  - alphas: {alphas.tolist()}")
    print(f"  - torques: {torques_arr.tolist()}")
    print(f"  - K (fixed): {k_fixed}")

    y0 = np.array([0.1, 0.0])
    t_eval = np.linspace(0, 10, 100)

    print(f"Initial condition: {y0.tolist()}")

    def pendulum_ode_scipy(
        t: float, y: np.ndarray, alpha: float, torque: float, k: float
    ) -> list[float]:
        theta, theta_dot = y[0], y[1]
        dtheta = theta_dot
        dtheta_dot = -alpha * theta_dot + torque - k * np.sin(theta)
        return [dtheta, dtheta_dot]

    print("\nRunning sequential solve_ivp...")
    start = time.perf_counter()
    results: list[np.ndarray] = []
    for i in range(n_params):
        sol = solve_ivp(  # type: ignore[no-untyped-call]
            fun=lambda t, y, a=alpha_flat[i], tr=torque_flat[i]: pendulum_ode_scipy(
                t, y, a, tr, k_fixed
            ),
            t_span=(0.0, 10.0),
            y0=y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8,
        )
        results.append(sol.y.T)  # (n_steps, n_dims)
    run_time = time.perf_counter() - start

    trajectories = np.stack(results, axis=0)  # (n_param_sets, n_steps, n_dims)

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}")
    print("  - (n_param_sets, n_timesteps, state_dim)")

    print("\nSample final states (first 5 parameter sets):")
    for i in range(5):
        final_state = trajectories[i, -1, :]
        print(
            f"  params[{i}] (\u03b1={alpha_flat[i]:.2f}, T={torque_flat[i]:.2f}): "
            f"\u03b8={final_state[0]:.4f}, \u03b8\u0307={final_state[1]:.4f}"
        )


def compare_results() -> None:
    """Compare results from both backends."""
    print("\n" + "=" * 60)
    print("Comparing Results (JAX vs PyTorch)")
    print("=" * 60)

    alphas_jax = jnp.linspace(0.1, 1.0, 3)
    torques_jax = jnp.linspace(0.0, 1.0, 3)
    k_fixed = 1.0

    alpha_grid_jax, torque_grid_jax = jnp.meshgrid(alphas_jax, torques_jax, indexing="ij")
    param_grid_jax = jnp.stack(
        [alpha_grid_jax.flatten(), torque_grid_jax.flatten(), jnp.full(9, k_fixed)],
        axis=1,
    )

    def pendulum_ode_jax(
        t: Any, y: jax.Array, args: tuple[jax.Array, jax.Array, jax.Array]
    ) -> jax.Array:
        alpha, torque, k = args
        theta, theta_dot = y[0], y[1]
        dtheta = theta_dot
        dtheta_dot = -alpha * theta_dot + torque - k * jnp.sin(theta)
        return jnp.array([dtheta, dtheta_dot])

    def solve_single_jax(params: jax.Array) -> jax.Array:
        alpha, torque, k = params[0], params[1], params[2]
        term = ODETerm(lambda t, y, _: pendulum_ode_jax(t, y, (alpha, torque, k)))
        saveat = SaveAt(ts=jnp.linspace(0, 10, 50))
        sol = diffeqsolve(
            term,
            Tsit5(),
            t0=0.0,
            t1=10.0,
            dt0=0.01,
            y0=jnp.array([0.1, 0.0]),
            saveat=saveat,
        )
        assert isinstance(sol.ys, jax.Array)
        return sol.ys

    batched_solve_jax = jax.vmap(solve_single_jax)
    traj_jax = batched_solve_jax(param_grid_jax)

    alphas_torch = torch.linspace(0.1, 1.0, 3, dtype=torch.float64)
    torques_torch = torch.linspace(0.0, 1.0, 3, dtype=torch.float64)

    alpha_grid_torch, torque_grid_torch = torch.meshgrid(alphas_torch, torques_torch, indexing="ij")
    alpha_flat = alpha_grid_torch.flatten()
    torque_flat = torque_grid_torch.flatten()
    k_flat = torch.full((9,), k_fixed, dtype=torch.float64)

    y0 = torch.tensor([[0.1, 0.0]], dtype=torch.float64).expand(9, -1).clone()
    ode_torch = BatchedPendulumTorch(alpha_flat, torque_flat, k_flat)
    t_span = torch.linspace(0, 10, 50, dtype=torch.float64)
    traj_torch_result = odeint(ode_torch, y0, t_span, method="dopri5")
    assert isinstance(traj_torch_result, torch.Tensor)
    traj_torch = traj_torch_result.permute(1, 0, 2)

    # --- torchode ---
    ode_torchode = BatchedPendulumTorch(alpha_flat, torque_flat, k_flat)
    n_compare_steps = 50
    t_eval_torchode = torch.linspace(0, 10, n_compare_steps, dtype=torch.float64)
    t_eval_batched = t_eval_torchode.unsqueeze(0).expand(9, -1)

    term_to = to.ODETerm(ode_torchode)
    step_method_to = to.Dopri5(term=term_to)
    controller_to = to.IntegralController(atol=1e-6, rtol=1e-8, term=term_to)
    solver_to = to.AutoDiffAdjoint(step_method_to, controller_to)  # pyright: ignore[reportArgumentType]
    problem_to = to.InitialValueProblem(
        y0=y0,  # pyright: ignore[reportArgumentType]
        t_start=torch.full((9,), 0.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
        t_end=torch.full((9,), 10.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
        t_eval=t_eval_batched,  # pyright: ignore[reportArgumentType]
    )
    solution_to = solver_to.solve(problem_to)
    traj_torchode = solution_to.ys  # (batch, n_steps, n_dims)

    # --- scipy ---
    t_eval_scipy = np.linspace(0, 10, n_compare_steps)
    scipy_results: list[np.ndarray] = []
    for i in range(9):
        a, tr, kv = float(alpha_flat[i]), float(torque_flat[i]), float(k_flat[i])
        sol = solve_ivp(  # type: ignore[no-untyped-call]
            fun=lambda t, y, _a=a, _tr=tr, _kv=kv: [
                y[1],
                -_a * y[1] + _tr - _kv * np.sin(y[0]),
            ],
            t_span=(0.0, 10.0),
            y0=[0.1, 0.0],
            method="RK45",
            t_eval=t_eval_scipy,
            rtol=1e-8,
            atol=1e-8,
        )
        scipy_results.append(sol.y.T)
    traj_scipy = np.stack(scipy_results, axis=0)  # (batch, n_steps, n_dims)

    # --- compare all pairs against JAX reference ---
    traj_jax_np = jax.device_get(traj_jax)
    traj_torch_np = traj_torch.numpy()
    traj_torchode_np = traj_torchode.detach().numpy()

    pairs: list[tuple[str, np.ndarray]] = [
        ("torchdiffeq", traj_torch_np),
        ("torchode", traj_torchode_np),
        ("scipy", traj_scipy),
    ]

    for name, traj in pairs:
        diff = abs(traj_jax_np - traj)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"\nJAX vs {name}:")
        print(f"  Max absolute difference:  {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        if max_diff < 1e-4:
            print("  \u2713 Results match within tolerance!")
        else:
            print("  \u26a0 Results differ significantly - check solver settings")


# ---------------------------------------------------------------------------
# Joint IC + parameter batching  ->  output shape (P, B, N, S)
#
# P = number of parameter sets
# B = number of initial conditions  (identical set reused for every P)
# N = number of timesteps
# S = state dimension
# ---------------------------------------------------------------------------


def run_diffrax_joint_batching() -> None:
    """Test joint IC + parameter batching with Diffrax. Output shape: (P, B, N, S)."""
    print("=" * 60)
    print("Diffrax Joint IC + Parameter Batching  ->  (P, B, N, S)")
    print("=" * 60)

    def solve_single(params: jax.Array, y0: jax.Array) -> jax.Array:
        alpha, torque, k = params[0], params[1], params[2]

        def ode(t: Any, y: jax.Array, _: None) -> jax.Array:
            return jnp.array([y[1], -alpha * y[1] + torque - k * jnp.sin(y[0])])

        saveat = SaveAt(ts=jnp.linspace(0, 10, 100))
        sol = diffeqsolve(ODETerm(ode), Tsit5(), t0=0.0, t1=10.0, dt0=0.01, y0=y0, saveat=saveat)
        assert isinstance(sol.ys, jax.Array)
        return sol.ys  # (N, S)

    # Single flat vmap over P*B trajectories (faster than nested double-vmap)
    solve_batch = jax.vmap(solve_single)

    ag, tg = jnp.meshgrid(jnp.linspace(0.1, 1.0, 3), jnp.linspace(0.0, 2.0, 3), indexing="ij")
    param_grid = jnp.stack([ag.flatten(), tg.flatten(), jnp.full(9, 1.0)], axis=1)  # (P=9, 3)

    thg, tdg = jnp.meshgrid(jnp.linspace(-1.0, 1.0, 3), jnp.linspace(-0.5, 0.5, 3), indexing="ij")
    y0_batch = jnp.stack([thg.flatten(), tdg.flatten()], axis=1)  # (B=9, 2)

    P, B = param_grid.shape[0], y0_batch.shape[0]
    print(f"P={P} parameter sets, B={B} initial conditions (shared across all P)")

    # Flatten (P, B) -> (P*B,): row p*B+b carries params[p] and IC[b]
    params_flat = jnp.repeat(param_grid, B, axis=0)  # (P*B, 3)
    y0_flat = jnp.tile(y0_batch, (P, 1))  # (P*B, 2)

    print(f"Flattened batch: P*B={P * B}")
    print("\nCompiling (first run)...")
    start = time.perf_counter()
    out_flat = solve_batch(params_flat, y0_flat)
    out_flat.block_until_ready()
    print(f"Compile + first run: {time.perf_counter() - start:.3f}s")

    start = time.perf_counter()
    out_flat = solve_batch(params_flat, y0_flat)
    out_flat.block_until_ready()
    print(f"Compiled run: {time.perf_counter() - start:.3f}s")

    # Reshape (P*B, N, S) -> (P, B, N, S)
    N_steps = out_flat.shape[1]
    trajectories = out_flat.reshape(P, B, N_steps, 2)
    print(f"Output shape: {trajectories.shape}  (P, B, N, S)")

    print("\nFinal states for IC b=0 across all parameter sets (same starting point):")
    for p in range(P):
        final = trajectories[p, 0, -1, :]
        print(
            f"  p={p:2d}  alpha={float(param_grid[p, 0]):.2f}  T={float(param_grid[p, 1]):.2f}: "
            f"theta={float(final[0]):.4f}  theta_dot={float(final[1]):.4f}"
        )


def run_torchdiffeq_joint_batching() -> None:
    """Test joint IC + parameter batching with torchdiffeq. Output shape: (P, B, N, S)."""
    print("\n" + "=" * 60)
    print("torchdiffeq Joint IC + Parameter Batching  ->  (P, B, N, S)")
    print("=" * 60)

    ag, tg = torch.meshgrid(
        torch.linspace(0.1, 1.0, 3, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        indexing="ij",
    )
    alpha_flat = ag.flatten()  # (P=9,)
    torque_flat = tg.flatten()
    k_flat = torch.full((9,), 1.0, dtype=torch.float64)

    thg, tdg = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 3, dtype=torch.float64),
        torch.linspace(-0.5, 0.5, 3, dtype=torch.float64),
        indexing="ij",
    )
    y0_batch = torch.stack([thg.flatten(), tdg.flatten()], dim=1)  # (B=9, 2)

    P, B = alpha_flat.shape[0], y0_batch.shape[0]
    print(f"P={P} parameter sets, B={B} initial conditions (shared across all P)")

    # Flatten to (P*B,): row p*B+b carries params[p] and IC[b]
    alpha_exp = alpha_flat.unsqueeze(1).expand(P, B).reshape(P * B)
    torque_exp = torque_flat.unsqueeze(1).expand(P, B).reshape(P * B)
    k_exp = k_flat.unsqueeze(1).expand(P, B).reshape(P * B)
    y0_exp = y0_batch.unsqueeze(0).expand(P, B, 2).reshape(P * B, 2).clone()

    ode = BatchedPendulumTorch(alpha_exp, torque_exp, k_exp)
    t_span = torch.linspace(0, 10, 100, dtype=torch.float64)

    print(f"Flattened batch: P*B={P * B}")
    print("\nRunning batched solve...")
    start = time.perf_counter()
    result = odeint(ode, y0_exp, t_span, method="dopri5")
    assert isinstance(result, torch.Tensor)
    run_time = time.perf_counter() - start

    # result: (N, P*B, S) -> (P, B, N, S)
    N_steps, _, S = result.shape
    trajectories = result.permute(1, 0, 2).reshape(P, B, N_steps, S)

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}  (P, B, N, S)")

    print("\nFinal states for IC b=0 across all parameter sets (same starting point):")
    for p in range(P):
        final = trajectories[p, 0, -1, :]
        print(
            f"  p={p:2d}  alpha={float(alpha_flat[p]):.2f}  T={float(torque_flat[p]):.2f}: "
            f"theta={float(final[0]):.4f}  theta_dot={float(final[1]):.4f}"
        )


def run_torchode_joint_batching() -> None:
    """Test joint IC + parameter batching with torchode. Output shape: (P, B, N, S)."""
    print("\n" + "=" * 60)
    print("torchode Joint IC + Parameter Batching  ->  (P, B, N, S)")
    print("=" * 60)

    ag, tg = torch.meshgrid(
        torch.linspace(0.1, 1.0, 3, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        indexing="ij",
    )
    alpha_flat = ag.flatten()
    torque_flat = tg.flatten()
    k_flat = torch.full((9,), 1.0, dtype=torch.float64)

    thg, tdg = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 3, dtype=torch.float64),
        torch.linspace(-0.5, 0.5, 3, dtype=torch.float64),
        indexing="ij",
    )
    y0_batch = torch.stack([thg.flatten(), tdg.flatten()], dim=1)  # (B=9, 2)

    P, B = alpha_flat.shape[0], y0_batch.shape[0]
    print(f"P={P} parameter sets, B={B} initial conditions (shared across all P)")

    alpha_exp = alpha_flat.unsqueeze(1).expand(P, B).reshape(P * B)
    torque_exp = torque_flat.unsqueeze(1).expand(P, B).reshape(P * B)
    k_exp = k_flat.unsqueeze(1).expand(P, B).reshape(P * B)
    y0_exp = y0_batch.unsqueeze(0).expand(P, B, 2).reshape(P * B, 2).clone()

    n_steps = 100
    t_eval = torch.linspace(0, 10, n_steps, dtype=torch.float64)
    t_eval_batched = t_eval.unsqueeze(0).expand(P * B, -1)

    ode = BatchedPendulumTorch(alpha_exp, torque_exp, k_exp)
    term = to.ODETerm(ode)
    step_method = to.Dopri5(term=term)
    step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-8, term=term)
    solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # pyright: ignore[reportArgumentType]

    problem = to.InitialValueProblem(
        y0=y0_exp,  # pyright: ignore[reportArgumentType]
        t_start=torch.full((P * B,), 0.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
        t_end=torch.full((P * B,), 10.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
        t_eval=t_eval_batched,  # pyright: ignore[reportArgumentType]
    )

    print(f"Flattened batch: P*B={P * B}")
    print("\nRunning batched solve...")
    start = time.perf_counter()
    solution = solver.solve(problem)
    run_time = time.perf_counter() - start

    # solution.ys: (P*B, N, S) -> (P, B, N, S)
    flat_traj = solution.ys
    N_steps, S = flat_traj.shape[1], flat_traj.shape[2]
    trajectories = flat_traj.reshape(P, B, N_steps, S)

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}  (P, B, N, S)")

    print("\nFinal states for IC b=0 across all parameter sets (same starting point):")
    for p in range(P):
        final = trajectories[p, 0, -1, :]
        print(
            f"  p={p:2d}  alpha={float(alpha_flat[p]):.2f}  T={float(torque_flat[p]):.2f}: "
            f"theta={float(final[0]):.4f}  theta_dot={float(final[1]):.4f}"
        )


def run_scipy_joint_batching() -> None:
    """Test joint IC + parameter batching with SciPy. Output shape: (P, B, N, S)."""
    print("\n" + "=" * 60)
    print("SciPy Joint IC + Parameter Batching  ->  (P, B, N, S)")
    print("=" * 60)

    ag, tg = np.meshgrid(np.linspace(0.1, 1.0, 3), np.linspace(0.0, 2.0, 3), indexing="ij")
    alpha_flat = ag.flatten()
    torque_flat = tg.flatten()
    k_fixed = 1.0

    thg, tdg = np.meshgrid(np.linspace(-1.0, 1.0, 3), np.linspace(-0.5, 0.5, 3), indexing="ij")
    y0_theta = thg.flatten()
    y0_theta_dot = tdg.flatten()

    P, B = alpha_flat.shape[0], y0_theta.shape[0]
    t_eval = np.linspace(0, 10, 100)

    print(f"P={P} parameter sets, B={B} initial conditions (shared across all P)")
    print(f"Running {P * B} sequential solve_ivp calls...")

    start = time.perf_counter()
    rows: list[list[np.ndarray]] = []
    for p in range(P):
        row: list[np.ndarray] = []
        for b in range(B):
            a, tr = float(alpha_flat[p]), float(torque_flat[p])
            sol = solve_ivp(  # type: ignore[no-untyped-call]
                fun=lambda t, y, _a=a, _tr=tr: [y[1], -_a * y[1] + _tr - k_fixed * np.sin(y[0])],
                t_span=(0.0, 10.0),
                y0=[float(y0_theta[b]), float(y0_theta_dot[b])],
                method="RK45",
                t_eval=t_eval,
                rtol=1e-8,
                atol=1e-8,
            )
            row.append(sol.y.T)  # (N, S)
        rows.append(row)
    run_time = time.perf_counter() - start

    trajectories = np.array([[rows[p][b] for b in range(B)] for p in range(P)])  # (P, B, N, S)

    print(f"Run time: {run_time:.3f}s")
    print(f"Output shape: {trajectories.shape}  (P, B, N, S)")

    print("\nFinal states for IC b=0 across all parameter sets (same starting point):")
    for p in range(P):
        final = trajectories[p, 0, -1, :]
        print(
            f"  p={p:2d}  alpha={alpha_flat[p]:.2f}  T={torque_flat[p]:.2f}: "
            f"theta={final[0]:.4f}  theta_dot={final[1]:.4f}"
        )


def compare_joint_results() -> None:
    """Compare joint IC + parameter batch results across all backends."""
    print("\n" + "=" * 60)
    print("Comparing Joint Results (P, B, N, S) Across Backends")
    print("=" * 60)

    # Small grids for fast comparison: P=4 (2x2), B=4 (2x2)
    alpha_np = np.array([0.1, 1.0, 0.1, 1.0])
    torque_np = np.array([0.0, 0.0, 2.0, 2.0])
    P = 4
    k_fixed = 1.0
    y0_theta_np = np.array([-1.0, 1.0, -1.0, 1.0])
    y0_dot_np = np.array([-0.5, -0.5, 0.5, 0.5])
    B = 4
    n_steps = 50
    t_eval_np = np.linspace(0, 10, n_steps)

    # --- JAX (flat vmap) ---
    param_grid_jax = jnp.stack(
        [jnp.array(alpha_np), jnp.array(torque_np), jnp.full(P, k_fixed)], axis=1
    )
    y0_batch_jax = jnp.stack([jnp.array(y0_theta_np), jnp.array(y0_dot_np)], axis=1)

    def solve_single_jax(params: jax.Array, y0: jax.Array) -> jax.Array:
        alpha, torque, k = params[0], params[1], params[2]

        def ode(t: Any, y: jax.Array, _: None) -> jax.Array:
            return jnp.array([y[1], -alpha * y[1] + torque - k * jnp.sin(y[0])])

        saveat = SaveAt(ts=jnp.linspace(0, 10, n_steps))
        sol = diffeqsolve(ODETerm(ode), Tsit5(), t0=0.0, t1=10.0, dt0=0.01, y0=y0, saveat=saveat)
        assert isinstance(sol.ys, jax.Array)
        return sol.ys

    params_flat_jax = jnp.repeat(param_grid_jax, B, axis=0)  # (P*B, 3)
    y0_flat_jax = jnp.tile(y0_batch_jax, (P, 1))  # (P*B, 2)
    out_flat_jax = jax.vmap(solve_single_jax)(params_flat_jax, y0_flat_jax)  # (P*B, N, S)
    traj_jax_np: np.ndarray = jax.device_get(out_flat_jax.reshape(P, B, n_steps, 2))  # (P, B, N, S)

    # --- torchdiffeq ---
    alpha_t = torch.tensor(alpha_np)
    torque_t = torch.tensor(torque_np)
    k_t = torch.full((P,), k_fixed, dtype=torch.float64)
    y0_t = torch.stack([torch.tensor(y0_theta_np), torch.tensor(y0_dot_np)], dim=1)

    alpha_exp = alpha_t.unsqueeze(1).expand(P, B).reshape(P * B)
    torque_exp = torque_t.unsqueeze(1).expand(P, B).reshape(P * B)
    k_exp = k_t.unsqueeze(1).expand(P, B).reshape(P * B)
    y0_exp = y0_t.unsqueeze(0).expand(P, B, 2).reshape(P * B, 2).clone()

    ode_td = BatchedPendulumTorch(alpha_exp, torque_exp, k_exp)
    t_span_t = torch.linspace(0, 10, n_steps, dtype=torch.float64)
    result_td = odeint(ode_td, y0_exp, t_span_t, method="dopri5")
    assert isinstance(result_td, torch.Tensor)
    traj_td_np = result_td.permute(1, 0, 2).reshape(P, B, n_steps, 2).numpy()

    # --- torchode ---
    ode_to = BatchedPendulumTorch(alpha_exp, torque_exp, k_exp)
    t_eval_t = t_span_t.unsqueeze(0).expand(P * B, -1)
    term_to = to.ODETerm(ode_to)
    solver_to = to.AutoDiffAdjoint(  # pyright: ignore[reportArgumentType]
        to.Dopri5(term=term_to),
        to.IntegralController(atol=1e-6, rtol=1e-8, term=term_to),
    )
    sol_to = solver_to.solve(
        to.InitialValueProblem(
            y0=y0_exp,  # pyright: ignore[reportArgumentType]
            t_start=torch.full((P * B,), 0.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
            t_end=torch.full((P * B,), 10.0, dtype=torch.float64),  # pyright: ignore[reportArgumentType]
            t_eval=t_eval_t,  # pyright: ignore[reportArgumentType]
        )
    )
    traj_to_np = sol_to.ys.reshape(P, B, n_steps, 2).detach().numpy()

    # --- scipy ---
    traj_scipy = np.zeros((P, B, n_steps, 2))
    for p in range(P):
        for b in range(B):
            a, tr = float(alpha_np[p]), float(torque_np[p])
            sol = solve_ivp(  # type: ignore[no-untyped-call]
                fun=lambda t, y, _a=a, _tr=tr: [y[1], -_a * y[1] + _tr - k_fixed * np.sin(y[0])],
                t_span=(0.0, 10.0),
                y0=[float(y0_theta_np[b]), float(y0_dot_np[b])],
                method="RK45",
                t_eval=t_eval_np,
                rtol=1e-8,
                atol=1e-8,
            )
            traj_scipy[p, b] = sol.y.T

    # --- numerical comparison ---
    pairs: list[tuple[str, np.ndarray]] = [
        ("torchdiffeq", traj_td_np),
        ("torchode", traj_to_np),
        ("scipy", traj_scipy),
    ]
    for name, traj in pairs:
        diff = np.abs(traj_jax_np - traj)
        print(
            f"\nJAX vs {name}:  max={diff.max():.2e}  mean={diff.mean():.2e}  "
            + ("\u2713 match" if diff.max() < 1e-4 else "\u26a0 mismatch")
        )

    # Verify shared ICs: at t=0 every parameter set must start from the same state
    print("\nVerifying shared ICs (t=0 state identical across all P for each B):")
    for b in range(B):
        ic_states = traj_jax_np[:, b, 0, :]  # (P, 2) — should all equal y0[b]
        ic_match = np.allclose(ic_states, ic_states[0:1], atol=1e-12)
        print(
            f"  b={b}  IC=(theta={y0_theta_np[b]:.2f}, theta_dot={y0_dot_np[b]:.2f}): "
            f"all P identical at t=0? {'yes' if ic_match else 'NO'}"
        )


def benchmark_diffrax_scaling() -> None:
    """
    Show that JAX vmap cost depends only on total trajectories (P * B), not on how
    they are split between parameters and initial conditions.

    Hypothesis: (P=5, B=2000) ≈ (P=1, B=10000)  because both are 10 000 trajectories.

    Uses the same integration settings as setup_pendulum_system.py / JaxSolver:
    t_span=(0, 1000), t_steps=1000, t_eval=(950, 1000),
    Dopri5, dt0=None, PIDController(rtol=1e-8, atol=1e-6), max_steps=16**5.

    A fixed-param baseline (matching JaxSolver exactly, params as Python constants)
    is printed first for reference — this is the ~9.777 s that JaxSolver achieves.
    The param-batching rows show the overhead of tracing params as dynamic arrays.
    """
    print("\n\n" + "#" * 60)
    print("# Diffrax Scaling: P*B constant, vary P vs B")
    print("#" * 60)

    T0, T1, T_STEADY = 0.0, 1000.0, 950.0
    N_SAVE = 1000
    B_TOTAL = 10_000

    saveat = SaveAt(ts=jnp.linspace(T_STEADY, T1, N_SAVE))
    pid = PIDController(rtol=1e-8, atol=1e-6)

    # ------------------------------------------------------------------
    # Baseline: fixed params as Python constants (replicates JaxSolver)
    # ------------------------------------------------------------------
    ALPHA_BASE, T_BASE, K_BASE = 0.1, 0.5, 1.0

    def solve_fixed_params(y0: jax.Array) -> jax.Array:
        def ode(t: Any, y: jax.Array, _: None) -> jax.Array:
            return jnp.array([y[1], -ALPHA_BASE * y[1] + T_BASE - K_BASE * jnp.sin(y[0])])

        sol = diffeqsolve(
            ODETerm(ode),
            Dopri5(),
            t0=T0,
            t1=T1,
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=pid,
            max_steps=16**5,
        )
        assert isinstance(sol.ys, jax.Array)
        return sol.ys

    solve_fixed_batch = jax.jit(jax.vmap(solve_fixed_params))

    thetas = jnp.linspace(-jnp.pi, jnp.pi, B_TOTAL)
    y0_base = jnp.stack([thetas, jnp.zeros(B_TOTAL)], axis=1)

    print(f"\n{'mode':<22}  {'P':>5}  {'B':>7}  {'P*B':>7}  {'time (s)':>10}  shape")
    print("-" * 72)

    start = time.perf_counter()
    out_fixed = solve_fixed_batch(y0_base)
    out_fixed.block_until_ready()
    t_fixed = time.perf_counter() - start
    print(
        f"{'fixed params (baseline)':<22}  {'1':>5}  {B_TOTAL:>7}  {B_TOTAL:>7}  {t_fixed:>10.3f}  {out_fixed.shape}"
    )

    # ------------------------------------------------------------------
    # Param-batching: params as dynamic traced arrays via vmap
    # ------------------------------------------------------------------
    def make_inputs(p: int, b: int) -> tuple[jax.Array, jax.Array]:
        """Flatten (P, B) -> (P*B,): row p*B+b carries params[p] and IC[b]."""
        alphas = jnp.linspace(0.1, 1.0, max(p, 2))[:p]
        param_grid = jnp.stack([alphas, jnp.ones(p) * 0.5, jnp.ones(p)], axis=1)  # (P, 3)
        thetas_b = jnp.linspace(-jnp.pi, jnp.pi, b)
        y0_batch = jnp.stack([thetas_b, jnp.zeros(b)], axis=1)  # (B, 2)
        params_flat = jnp.repeat(param_grid, b, axis=0)  # (P*B, 3)
        y0_flat = jnp.tile(y0_batch, (p, 1))  # (P*B, 2)
        return params_flat, y0_flat

    # Defined once — JAX reuses the compiled kernel for all (P, B) splits
    # because the leading dimension P*B=10000 stays constant.
    def solve_one(params: jax.Array, y0: jax.Array) -> jax.Array:
        alpha, torque, k = params[0], params[1], params[2]

        def ode(t: Any, y: jax.Array, _: None) -> jax.Array:
            return jnp.array([y[1], -alpha * y[1] + torque - k * jnp.sin(y[0])])

        sol = diffeqsolve(
            ODETerm(ode),
            Dopri5(),
            t0=T0,
            t1=T1,
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=pid,
            max_steps=16**5,
        )
        assert isinstance(sol.ys, jax.Array)
        return sol.ys

    solve_batch = jax.jit(jax.vmap(solve_one))

    cases: list[tuple[int, int]] = [
        (1, 10_000),
        (5, 2_000),
        (10, 1_000),
        (50, 200),
        (100, 100),
    ]

    for p, b in cases:
        params_flat, y0_flat = make_inputs(p, b)
        start = time.perf_counter()
        out_flat = solve_batch(params_flat, y0_flat)  # (P*B, N_SAVE, S)
        out_flat.block_until_ready()
        elapsed = time.perf_counter() - start
        out = out_flat.reshape(p, b, N_SAVE, 2)  # (P, B, N_SAVE, S)
        print(f"{'dynamic params':<22}  {p:>5}  {b:>7}  {p * b:>7}  {elapsed:>10.3f}  {out.shape}")


def benchmark_torchdiffeq_scaling() -> None:
    """
    Show that torchdiffeq cost depends only on P*B, not on split.

    Same integration settings as benchmark_diffrax_scaling:
    t_span=(0, 1000), steady-state window (950, 1000), N_SAVE=1000,
    Dopri5, rtol=1e-8, atol=1e-6.
    B_TOTAL=4000.
    """
    print("\n\n" + "#" * 60)
    print("# torchdiffeq Scaling: P*B constant, vary P vs B")
    print("#" * 60)

    T0, T1, T_STEADY = 0.0, 1000.0, 950.0
    N_SAVE = 1000
    B_TOTAL = 4000

    # odeint requires t[0] == t0; include T0, then record the steady-state window
    t_eval = torch.cat(
        [
            torch.tensor([T0], dtype=torch.float64),
            torch.linspace(T_STEADY, T1, N_SAVE, dtype=torch.float64),
        ]
    )

    print(f"\n{'mode':<22}  {'P':>5}  {'B':>7}  {'P*B':>7}  {'time (s)':>10}  shape")
    print("-" * 72)

    cases: list[tuple[int, int]] = [
        (1, B_TOTAL),
        (4, B_TOTAL // 4),
        (10, B_TOTAL // 10),
        (20, B_TOTAL // 20),
        (40, B_TOTAL // 40),
    ]

    for p, b in cases:
        alphas = torch.linspace(0.1, 1.0, p, dtype=torch.float64)
        alphas_exp = alphas.unsqueeze(1).expand(p, b).reshape(p * b)
        torques_exp = torch.full((p * b,), 0.5, dtype=torch.float64)
        ks_exp = torch.full((p * b,), 1.0, dtype=torch.float64)

        thetas = torch.linspace(-float(np.pi), float(np.pi), b, dtype=torch.float64)
        y0_b = torch.stack([thetas, torch.zeros(b, dtype=torch.float64)], dim=1)
        y0_flat = y0_b.unsqueeze(0).expand(p, b, 2).reshape(p * b, 2).clone()

        ode = BatchedPendulumTorch(alphas_exp, torques_exp, ks_exp)

        start = time.perf_counter()
        result = odeint(ode, y0_flat, t_eval, method="dopri5", rtol=1e-8, atol=1e-6)
        assert isinstance(result, torch.Tensor)
        elapsed = time.perf_counter() - start

        # result: (N_SAVE+1, P*B, 2); drop t=T0 -> (N_SAVE, P*B, 2) -> (P, B, N_SAVE, 2)
        trajectories = result[1:].permute(1, 0, 2).reshape(p, b, N_SAVE, 2)
        print(
            f"{'dynamic params':<22}  {p:>5}  {b:>7}  {p * b:>7}  {elapsed:>10.3f}  {trajectories.shape}"
        )


def benchmark_scipy_scaling() -> None:
    """
    Show that scipy (sequential) cost depends only on P*B.

    Uses t_span=(0, 10) to keep sequential runtime manageable — the P*B=constant
    hypothesis is trivially true for sequential solvers regardless of t_span.
    B_TOTAL=100.
    """
    print("\n\n" + "#" * 60)
    print("# SciPy Scaling: P*B constant, vary P vs B")
    print("# (sequential solve_ivp; t_span=(0, 10) to keep runtime short)")
    print("#" * 60)

    T0, T1 = 0.0, 10.0
    N_SAVE = 100
    B_TOTAL = 100

    t_eval_np = np.linspace(T0, T1, N_SAVE)

    print(f"\n{'mode':<22}  {'P':>5}  {'B':>7}  {'P*B':>7}  {'time (s)':>10}  shape")
    print("-" * 72)

    cases: list[tuple[int, int]] = [
        (1, B_TOTAL),
        (4, B_TOTAL // 4),
        (10, B_TOTAL // 10),
        (20, B_TOTAL // 20),
        (50, B_TOTAL // 50),
    ]

    for p, b in cases:
        alphas_p = np.linspace(0.1, 1.0, p)
        alphas_flat = np.repeat(alphas_p, b)
        torques_flat = np.full(p * b, 0.5)
        thetas = np.linspace(-np.pi, np.pi, b)
        y0_thetas = np.tile(thetas, p)
        y0_dots = np.zeros(p * b)

        start = time.perf_counter()
        results: list[np.ndarray] = []
        for i in range(p * b):
            a, tr = float(alphas_flat[i]), float(torques_flat[i])
            sol = solve_ivp(  # type: ignore[no-untyped-call]
                fun=lambda t, y, _a=a, _tr=tr: [y[1], -_a * y[1] + _tr - np.sin(y[0])],
                t_span=(T0, T1),
                y0=[float(y0_thetas[i]), float(y0_dots[i])],
                method="RK45",
                t_eval=t_eval_np,
                rtol=1e-8,
                atol=1e-6,
            )
            results.append(sol.y.T)
        elapsed = time.perf_counter() - start

        trajectories = np.array(results).reshape(p, b, N_SAVE, 2)
        print(
            f"{'dynamic params':<22}  {p:>5}  {b:>7}  {p * b:>7}  {elapsed:>10.3f}  {trajectories.shape}"
        )


if __name__ == "__main__":
    run_diffrax_parameter_batching()
    run_torchdiffeq_parameter_batching()
    run_torchode_parameter_batching()
    run_scipy_parameter_batching()
    compare_results()

    print("\n\n" + "#" * 60)
    print("# Joint IC + Parameter Batching  (P, B, N, S)")
    print("#" * 60)
    run_diffrax_joint_batching()
    run_torchdiffeq_joint_batching()
    run_torchode_joint_batching()
    run_scipy_joint_batching()
    compare_joint_results()

    benchmark_diffrax_scaling()
    benchmark_torchdiffeq_scaling()
    benchmark_scipy_scaling()
