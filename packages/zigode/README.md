# Zig ODE Solver

## System description

Tests and benchmarks use the driven damped pendulum as a reference problem:

$$
\begin{aligned}
\dot{\theta} &= \omega \\
\dot{\omega} &= -\alpha \omega - K \sin(\theta) + T
\end{aligned}
$$

### Parameters

| Parameter           | Symbol   | Value |
| ------------------- | -------- | ----- |
| Damping coefficient | $\alpha$ | 0.1   |
| Torque              | $T$      | 0.5   |
| Stiffness           | $K$      | 1.0   |

Under these parameter values the system has two coexisting attractors: a stable fixed point near $\theta^* = \arcsin(T/K) \approx 0.5236$ and a limit cycle corresponding to full rotational motion.

### Initial conditions

Five ICs span both basins of attraction:

| IC          | Attractor   |
| ----------- | ----------- |
| [0.4, 0.0]  | Fixed point |
| [1.0, 0.0]  | Fixed point |
| [2.7, 0.0]  | Limit cycle |
| [0.5, -5.0] | Limit cycle |
| [-2.0, 3.0] | Limit cycle |

### Solver settings

| Setting            | Value     |
| ------------------ | --------- |
| Method             | Dopri5    |
| Time span          | [0, 1000] |
| Steps (tests)      | 1000      |
| Steps (benchmarks) | 10,000    |
| Relative tolerance | 1e-8      |
| Absolute tolerance | 1e-6      |

Three independent Dopri5 implementations are compared: a hand-written Zig ODE compiled via `zigode`, a JAX/Diffrax backend with a PID step-size controller, and torchdiffeq with a classic controller. A fourth variant uses SymPy to generate C source code that compiles against the same Dopri5 engine as the Zig path.

## Test results

Results from `uv run python -m zigode.test_zig_solver`.

## Cross-solver trajectory errors

Point-by-point comparison across all 1000 time steps and both state variables.

### Zig vs JAX/Diffrax (PID step-size controller)

| IC          | Max abs err | Max rel err | Zig final theta | JAX final theta |
| ----------- | ----------- | ----------- | --------------- | --------------- |
| [0.4, 0.0]  | 5.37e-06    | 2.10e+00    | 0.523617        | 0.523614        |
| [2.7, 0.0]  | 3.01e-02    | 1.10e-03    | 4947.699671     | 4947.728516     |
| [0.5, -5.0] | 4.61e-02    | 1.50e-03    | 4904.608349     | 4904.652344     |
| [-2.0, 3.0] | 1.99e-02    | 4.90e-04    | 4975.221288     | 4975.236816     |
| [1.0, 0.0]  | 4.57e-06    | 8.38e-01    | 0.523583        | 0.523579        |

### Zig vs torchdiffeq (classic step-size controller)

| IC          | Max abs err | Max rel err | Zig final theta | TDE final theta |
| ----------- | ----------- | ----------- | --------------- | --------------- |
| [0.4, 0.0]  | 2.03e-05    | 1.90e+10    | 0.523617        | 0.523599        |
| [2.7, 0.0]  | 3.61e-03    | 1.39e-04    | 4947.699671     | 4947.703034     |
| [0.5, -5.0] | 6.10e-03    | 2.30e-04    | 4904.608349     | 4904.614031     |
| [-2.0, 3.0] | 1.77e-03    | 6.94e-05    | 4975.221288     | 4975.223032     |
| [1.0, 0.0]  | 1.74e-05    | 1.62e+10    | 0.523583        | 0.523599        |

### JAX vs torchdiffeq (cross-check, no Zig)

| IC          | Max abs err | Max rel err | JAX final theta | TDE final theta |
| ----------- | ----------- | ----------- | --------------- | --------------- |
| [0.4, 0.0]  | 1.82e-05    | 1.55e+10    | 0.523614        | 0.523599        |
| [2.7, 0.0]  | 2.65e-02    | 9.74e-04    | 4947.728516     | 4947.703034     |
| [0.5, -5.0] | 4.06e-02    | 1.28e-03    | 4904.652344     | 4904.614031     |
| [-2.0, 3.0] | 1.85e-02    | 4.37e-04    | 4975.236816     | 4975.223032     |
| [1.0, 0.0]  | 2.12e-05    | 1.94e+10    | 0.523579        | 0.523599        |

Fixed-point ICs ([0.4, 0.0] and [1.0, 0.0]) agree to ~1e-5 across all solver pairs. Limit-cycle ICs accumulate phase drift over the long integration span due to differing step-size controllers, reaching ~0.046 in the worst case (Zig vs JAX). The Zig-torchdiffeq pair shows tighter agreement (~6e-3) because both use a classic controller.

## Early-time check (t = 10)

At short integration times, all solvers agree to ~1e-5:

| IC          | Zig-JAX  | Zig-TDE  | JAX-TDE  |
| ----------- | -------- | -------- | -------- |
| [0.4, 0.0]  | 3.48e-07 | 2.97e-06 | 3.32e-06 |
| [2.7, 0.0]  | 2.47e-05 | 1.42e-05 | 1.04e-05 |
| [0.5, -5.0] | 4.35e-06 | 6.89e-07 | 5.04e-06 |
| [-2.0, 3.0] | 1.78e-06 | 1.51e-05 | 1.47e-05 |
| [1.0, 0.0]  | 3.17e-07 | 2.26e-06 | 2.58e-06 |

## SymPy codegen vs Zig ODE

Both use the same Dopri5 solver engine; differences come only from the C vs Zig compiler's `sin()` implementation.

| IC          | Max abs err |
| ----------- | ----------- |
| [0.4, 0.0]  | 2.00e-14    |
| [2.7, 0.0]  | 3.89e-05    |
| [0.5, -5.0] | 2.98e-05    |
| [-2.0, 3.0] | 3.90e-05    |
| [1.0, 0.0]  | 1.04e-14    |

Overall max: 3.90e-05.

## Test tolerances

| Constant                 | Value | Purpose                                      |
| ------------------------ | ----- | -------------------------------------------- |
| `CROSS_SOLVER_ATOL`      | 0.05  | Overall trajectory comparison across solvers |
| `FIXED_POINT_ATOL`       | 1e-3  | Fixed-point IC comparison                    |
| `LIMIT_CYCLE_OMEGA_ATOL` | 1e-2  | Final angular velocity on limit cycles       |
| `EARLY_TIME_ATOL`        | 1e-3  | Early-time cross-solver agreement            |
| `TIME_GRID_ATOL`         | 1e-10 | Time grid matching                           |
| `SYMPY_VS_ZIG_ATOL`      | 5e-5  | SymPy-generated C vs hand-written Zig ODE    |

## Test summary

36 passed, 0 failed. Performance: ~117 us/call (single IC, 1000 steps).

## Scaling benchmark

Results from `uv run python -m zigode.benchmark_zig_solver`.

Each N value was run 5 times with 10,000 output steps over `t in [0, 1000]`. The "min" column reports the fastest of the 5 rounds; "us/IC" is derived from that minimum.

### Zig ODE (pendulum.zig)

| N       | min (s) | mean (s) | std (s) | us/IC |
| ------- | ------- | -------- | ------- | ----- |
| 100     | 0.008   | 0.009    | 0.001   | 80.1  |
| 200     | 0.017   | 0.020    | 0.002   | 83.8  |
| 500     | 0.029   | 0.031    | 0.002   | 57.2  |
| 1,000   | 0.061   | 0.065    | 0.003   | 61.5  |
| 2,000   | 0.114   | 0.123    | 0.007   | 57.0  |
| 5,000   | 0.251   | 0.291    | 0.036   | 50.1  |
| 10,000  | 0.526   | 0.576    | 0.043   | 52.6  |
| 20,000  | 1.016   | 1.185    | 0.119   | 50.8  |
| 50,000  | 2.467   | 2.517    | 0.050   | 49.3  |
| 100,000 | 5.091   | 5.344    | 0.376   | 50.9  |

### SymPy C ODE (pendulum_sympy.c)

| N       | min (s) | mean (s) | std (s) | us/IC |
| ------- | ------- | -------- | ------- | ----- |
| 100     | 0.010   | 0.011    | 0.001   | 101.5 |
| 200     | 0.018   | 0.019    | 0.001   | 87.9  |
| 500     | 0.035   | 0.037    | 0.002   | 69.8  |
| 1,000   | 0.066   | 0.071    | 0.006   | 66.3  |
| 2,000   | 0.131   | 0.162    | 0.020   | 65.4  |
| 5,000   | 0.346   | 0.421    | 0.076   | 69.2  |
| 10,000  | 0.674   | 0.760    | 0.067   | 67.4  |
| 20,000  | 1.371   | 1.468    | 0.088   | 68.6  |
| 50,000  | 3.566   | 3.649    | 0.082   | 71.3  |
| 100,000 | 6.985   | 7.406    | 0.426   | 69.9  |

Both backends scale linearly with N. The Zig-compiled ODE saturates at ~50 us/IC while the SymPy-generated C ODE settles around ~70 us/IC -- a 1.4x difference attributable to the Zig compiler producing tighter machine code for the RHS evaluation.
