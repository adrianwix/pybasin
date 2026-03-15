# pyright: basic
"""
Experiment: JAX JIT compilation correctness check for diffrax.

Results (64 ICs, 500 t_steps, t1=100, n=10 runs averaged):

  Scenario 1 — vmap only, cached outside run:
    call 1 (compile): 0.613s  |  call 2 (compiled): 0.0047s  |  speedup: 130×
    JAX trace-cache reuses XLA for stable module-level functions even without jit.

  Scenario 2 — jit(vmap) cached outside run:
    call 1 (compile): 0.600s  |  call 2 (compiled): 0.0038s  |  speedup: 159×
    Explicit jit gives marginally faster compiled calls than trace-cache alone.

  Scenario 3 — vmap recreated each call (mirrors JaxSolver._integrate_jax_generic):
    call 1 (compile): 0.639s  |  call 2 (compiled): 0.0049s  |  speedup: 131×
    NOT the bug: JAX still reuses XLA because the underlying ODE fn is stable.

  Scenario 4 — jit(vmap) recreated each call (hypothetical misuse, not JaxSolver):
    call 1 (compile): 0.526s  |  call 2 (compiled): 0.169s   |  speedup:   3×
    MISUSE: wrapping a new jit object each call forces a full retrace, defeating
    the trace cache. Call 2 is 44× slower than scenarios 1-2.

Conclusion:
  JaxSolver._integrate_jax_generic matches Scenario 3 (vmap recreated, no jit).
  There is NO performance bug: JAX's trace cache reuses XLA because the underlying
  ODE fn is a stable Python object. Scenario 4 shows what would go wrong if jit()
  were applied to a new object on each call — but JaxSolver never does this.


Diagnoses why JIT speedup is negligible by comparing four scenarios:
1. vmap only (no jit) — cached outside run, re-traces on first call then fast
2. jit(vmap) cached outside run — compiles on first call, fast on second
3. vmap recreated each call — mirrors JaxSolver; NOT a bug: JAX trace-cache works
4. jit(vmap) recreated each call — hypothetical misuse; new object defeats JIT cache

Each scenario calls the solve function TWICE.
Call 1 = compile + run (slow).
Call 2 = compiled path (should be fast for scenarios with proper JIT).
"""

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from diffrax._custom_types import RealScalarLike

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])

N_IC = 64  # batch size (initial conditions)
T_STEPS = 500
T1 = 100.0
ALPHA = 0.3
TORQUE = 0.5
K = 1.0


# ---------------------------------------------------------------------------
# Shared ODE definition (module-level so it is a stable Python object)
# ---------------------------------------------------------------------------
def pendulum_ode(t: RealScalarLike, y: jax.Array, args: None) -> jax.Array:
    theta, theta_dot = y[0], y[1]
    dtheta_dot = -ALPHA * theta_dot + TORQUE - K * jnp.sin(theta)
    return jnp.array([theta_dot, dtheta_dot])


def _make_y0(n: int) -> jax.Array:
    """Random initial conditions so each run uses different data (same shape)."""
    return jnp.stack(
        [
            jnp.linspace(-jnp.pi, jnp.pi, n),
            jnp.zeros(n),
        ],
        axis=1,
    )


def _solve_single(y0_single: jax.Array, save_ts: jax.Array) -> jax.Array:
    term = ODETerm(pendulum_ode)
    sol = diffeqsolve(
        term,
        Dopri5(),
        t0=0.0,
        t1=T1,
        dt0=None,
        y0=y0_single,
        saveat=SaveAt(ts=save_ts),
        stepsize_controller=PIDController(rtol=1e-8, atol=1e-6),
        max_steps=16**4,
    )
    return sol.ys  # type: ignore[return-value]


N_RUNS = 10


def _time_scenario(label: str, fn: Callable[[jax.Array], jax.Array], y0: jax.Array) -> None:
    """Run the full scenario N_RUNS times. Each run: clear cache, call 1 (compile), call 2 (compiled)."""
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")

    compile_times: list[float] = []
    compiled_times: list[float] = []

    for _ in range(N_RUNS):
        jax.clear_caches()

        start = time.perf_counter()
        result = fn(y0)
        jax.block_until_ready(result)
        compile_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        result = fn(y0)
        jax.block_until_ready(result)
        compiled_times.append(time.perf_counter() - start)

    avg_compile = sum(compile_times) / N_RUNS
    avg_compiled = sum(compiled_times) / N_RUNS
    speedup = avg_compile / avg_compiled if avg_compiled > 0 else float("inf")
    print(f"  call 1 avg (compile + run, n={N_RUNS}): {avg_compile:.3f}s")
    print(f"  call 2 avg (compiled,      n={N_RUNS}): {avg_compiled:.4f}s")
    print(f"  speedup avg1 / avg2:                    {speedup:.1f}×")


# ---------------------------------------------------------------------------
# Scenario 1 — vmap only, no jit
# ---------------------------------------------------------------------------
def scenario_vmap_no_jit() -> None:
    save_ts = jnp.linspace(0.0, T1, T_STEPS)
    y0 = _make_y0(N_IC)

    # New vmap wrapper created once (mimics the LOCAL variable in _integrate_jax_generic)
    batched_solve = jax.vmap(lambda y: _solve_single(y, save_ts))

    def run(y0: jax.Array) -> jax.Array:
        return batched_solve(y0)

    _time_scenario("Scenario 1 — vmap only (no jit): expect ~same time both calls", run, y0)


# ---------------------------------------------------------------------------
# Scenario 2 — jit(vmap), single compile
# ---------------------------------------------------------------------------
def scenario_jit_vmap() -> None:
    save_ts = jnp.linspace(0.0, T1, T_STEPS)
    y0 = _make_y0(N_IC)

    batched_solve = jax.jit(jax.vmap(lambda y: _solve_single(y, save_ts)))

    def run(y0: jax.Array) -> jax.Array:
        return batched_solve(y0)

    _time_scenario("Scenario 2 — jit(vmap) cached: expect call 2 much faster", run, y0)


# ---------------------------------------------------------------------------
# Scenario 3 — vmap recreated inside the run (mirrors JaxSolver._integrate_jax_generic)
# ---------------------------------------------------------------------------
def scenario_vmap_recreated_inside_loop() -> None:
    """
    vmap recreated on every call, no jit.
    JAX's trace cache still kicks in because the underlying ODE fn is a stable
    module-level object → call 2 is just as fast as scenario 1 and 2.
    This is NOT the performance bug.
    """
    save_ts = jnp.linspace(0.0, T1, T_STEPS)
    y0 = _make_y0(N_IC)

    def run(y0: jax.Array) -> jax.Array:
        batched_solve = jax.vmap(lambda y: _solve_single(y, save_ts))
        return batched_solve(y0)

    _time_scenario(
        "Scenario 3 — vmap recreated each call (no jit): expect call 2 still fast",
        run,
        y0,
    )


# ---------------------------------------------------------------------------
# Scenario 4 — jit(vmap) created inside the loop (hypothetical misuse)
# ---------------------------------------------------------------------------
def scenario_jit_vmap_recreated_inside_loop() -> None:
    """
    Hypothetical misuse not present in JaxSolver: creating a *new* jit-wrapped
    function on every call forces a full retrace because the compiled artifact
    is keyed to the Python function object.
    """
    save_ts = jnp.linspace(0.0, T1, T_STEPS)
    y0 = _make_y0(N_IC)

    def run(y0: jax.Array) -> jax.Array:
        # new jit(vmap) every call — still bad
        batched_solve = jax.jit(jax.vmap(lambda y: _solve_single(y, save_ts)))
        return batched_solve(y0)

    _time_scenario(
        "Scenario 4 — jit(vmap) recreated each call (misuse): expect slow both calls",
        run,
        y0,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary() -> None:
    print(
        """
╔══════════════════════════════════════════════════════════════════╗
║  DIAGNOSIS SUMMARY                                               ║
╠══════════════════════════════════════════════════════════════════╣
║  Scenario 1 — vmap only cached:  fast after call 1 (no jit)     ║
║  Scenario 2 — jit(vmap) cached:  fast after call 1              ║
║  Scenario 3 — vmap in loop:      NOT the bug (trace-cache ok)   ║
║  Scenario 4 — jit(vmap) in loop: hypothetical misuse, not JaxSolver ║
╠══════════════════════════════════════════════════════════════════╣
║  CONCLUSION for jax_solver.py:                                   ║
║    _integrate_jax_generic matches Scenario 3: vmap recreated per ║
║    call, no jit. This is NOT a bug — JAX's trace cache reuses    ║
║    XLA because the ODE fn is a stable Python object. No fix      ║
║    needed: performance is equivalent to the cached jit(vmap).    ║
╚══════════════════════════════════════════════════════════════════╝
"""
    )


if __name__ == "__main__":
    print(f"Batch size: {N_IC} ICs | t_steps: {T_STEPS} | t1: {T1}")
    print(
        f"Each scenario: call 1 = compile+run, each run: clear cache, call 1 = compile, call 2 = compiled (n={N_RUNS} runs)"
    )
    scenario_vmap_no_jit()
    scenario_jit_vmap()
    scenario_vmap_recreated_inside_loop()
    scenario_jit_vmap_recreated_inside_loop()
    print_summary()
