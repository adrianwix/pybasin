# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportReturnType=false
"""Direct diffrax benchmark for pendulum basin stability computation."""

import time

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

N_SAMPLES = 10000
T0 = 0.0
T1 = 1000.0
T_STEADY = 900.0
N_SAVE_TOTAL = 10000
FP_THRESHOLD = 0.01
STEADY_IDX = int((T_STEADY - T0) / (T1 - T0) * (N_SAVE_TOTAL - 1) + 0.5)


def pendulum_ode(t: float, y: Float[Array, "2"], params: Float[Array, "3"]) -> Float[Array, "2"]:
    """
    Pendulum ODE: dθ/dt = θ̇, dθ̇/dt = -α·θ̇ + T - K·sin(θ)

    Args:
        t: time (unused, autonomous system)
        y: state [theta, theta_dot]
        params: [alpha, T, K] - damping, torque, stiffness

    Same equations as the Zig implementation for fair comparison.
    """
    theta = y[0]
    theta_dot = y[1]

    alpha = params[0]
    torque = params[1]
    k = params[2]

    dtheta_dt = theta_dot
    dtheta_dot_dt = -alpha * theta_dot + torque - k * jnp.sin(theta)

    return jnp.array([dtheta_dt, dtheta_dot_dt])


def classify_trajectory(sol: Float[Array, "time 2"]) -> Int[Array, ""]:
    """
    Classify trajectory as Fixed Point (0) or Limit Cycle (1).

    Uses only steady-state portion (t >= T_STEADY).
    delta = |max(theta_dot) - mean(theta_dot)| < threshold => FP, else LC
    """
    steady_sol = sol[STEADY_IDX:, 1]

    max_val = jnp.max(steady_sol)
    mean_val = jnp.mean(steady_sol)
    delta = jnp.abs(max_val - mean_val)

    return jnp.where(delta < FP_THRESHOLD, 0, 1)


def solve_single_ic(y0: Float[Array, "2"], params: Float[Array, "3"]) -> Float[Array, "time 2"]:
    """
    Solve the pendulum ODE for a single initial condition.

    Args:
        y0: initial state [theta, theta_dot]
        params: [alpha, T, K]

    Returns the full trajectory (shape: N_SAVE_TOTAL x 2).
    """
    term = diffrax.ODETerm(lambda t, y, args: pendulum_ode(t, y, params))
    solver = diffrax.Dopri5()

    save_at = diffrax.SaveAt(ts=jnp.linspace(T0, T1, N_SAVE_TOTAL))

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=T0,
        t1=T1,
        dt0=None,
        y0=y0,
        saveat=save_at,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-6),
        max_steps=1_000_000,
    )

    return sol.ys


def solve_and_classify_batch(
    ics: Float[Array, "batch 2"], params: Float[Array, "3"]
) -> Int[Array, "batch"]:
    """
    Solve and classify a batch of initial conditions.

    Args:
        ics: initial conditions (N_SAMPLES x 2)
        params: [alpha, T, K]

    Returns labels: 0 for FP, 1 for LC.
    """

    def process_one(y0: Float[Array, "2"]) -> Int[Array, ""]:
        trajectory = solve_single_ic(y0, params)
        return classify_trajectory(trajectory)

    return jax.vmap(process_one)(ics)


def generate_initial_conditions(
    params: Float[Array, "3"], seed: int = 42
) -> Float[Array, "batch 2"]:
    """
    Generate uniform random initial conditions for the pendulum.

    Args:
        params: [alpha, T, K]
        seed: random seed
    """
    key = jax.random.PRNGKey(seed)

    torque = params[1]
    k = params[2]

    offset = jnp.arcsin(torque / k)
    theta_min = -jnp.pi + offset
    theta_max = jnp.pi + offset

    key_theta, key_theta_dot = jax.random.split(key)

    theta = jax.random.uniform(key_theta, (N_SAMPLES,), minval=theta_min, maxval=theta_max)
    theta_dot = jax.random.uniform(key_theta_dot, (N_SAMPLES,), minval=-10.0, maxval=10.0)

    return jnp.stack([theta, theta_dot], axis=1)


def main() -> None:
    print("=" * 60)
    print("Diffrax Pendulum Basin Stability Benchmark")
    print("=" * 60)

    # Parameters: [alpha, T, K]
    params = jnp.array([0.1, 0.5, 1.0])

    print(f"\nParameters: alpha={params[0]}, T={params[1]}, K={params[2]}")
    print(f"Generating {N_SAMPLES} initial conditions...")
    ics = generate_initial_conditions(params)

    print(f"Saving {N_SAVE_TOTAL} points from t={T0:.0f} to t={T1:.0f}")
    print(f"Using t>={T_STEADY:.0f} for classification")
    print("Solver: Dopri5 (rtol=1e-8, atol=1e-6)")

    print("\n" + "-" * 60)
    print("First run: Compiling + Solving...")
    print("-" * 60)

    jit_fn = jax.jit(solve_and_classify_batch)

    start = time.perf_counter()
    labels = jit_fn(ics, params).block_until_ready()
    elapsed_first = time.perf_counter() - start

    elapsed_ms_first = elapsed_first * 1000
    us_per_ic_first = (elapsed_first * 1_000_000) / N_SAMPLES

    print(
        f"First run (with JIT compilation): {elapsed_ms_first:.1f} ms ({us_per_ic_first:.1f} μs per IC)"
    )

    print("\n" + "-" * 60)
    print("Second run: Already compiled...")
    print("-" * 60)

    start = time.perf_counter()
    labels = jit_fn(ics, params).block_until_ready()
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000
    us_per_ic = (elapsed * 1_000_000) / N_SAMPLES

    print(f"Second run (compiled): {elapsed_ms:.1f} ms ({us_per_ic:.1f} μs per IC)")
    print(f"Speedup from compilation: {elapsed_first / elapsed:.2f}x")

    fp_count = int(jnp.sum(labels == 0))
    lc_count = int(jnp.sum(labels == 1))

    fp_frac = fp_count / N_SAMPLES
    lc_frac = lc_count / N_SAMPLES

    print("\n" + "=" * 60)
    print("Basin Stability Results")
    print("=" * 60)
    print(f"  Fixed Point (FP): {fp_count:5} / {N_SAMPLES}  =  {fp_frac:.4f}")
    print(f"  Limit Cycle (LC): {lc_count:5} / {N_SAMPLES}  =  {lc_frac:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
