# Pendulum

## System Description

Driven damped pendulum:

$$
\begin{aligned}
\dot{\theta} &= \omega \\
\dot{\omega} &= -\alpha \omega - K \sin(\theta) + T
\end{aligned}
$$

### System Parameters

| Parameter           | Symbol   | Value |
| ------------------- | -------- | ----- |
| Damping coefficient | $\alpha$ | 0.1   |
| Torque              | $T$      | 0.5   |
| Stiffness           | $K$      | 1.0   |

### Sampling

- **Dimension**: $D = 2$
- **Sample size**: $N = 10000$
- **Distribution**: $\rho$ = Uniform
- **Region of interest**: $\mathcal{Q}(\theta, \omega) : [\psi - \pi, \psi + \pi] \times [-10, 10]$ where $\psi = \arcsin(T/K)$

### Solver

| Setting            | Value               |
| ------------------ | ------------------- |
| Method             | Dopri5 (Diffrax)    |
| Time span          | $[0, 1000]$         |
| Steps              | 1000 ($f_s$ = 1 Hz) |
| Relative tolerance | 1e-08               |
| Absolute tolerance | 1e-06               |

### Feature Extraction

Log-delta feature on angular velocity:

- States: $\omega$ (state 1)
- Formula: $\Delta = \log_{10}(|\max(\omega) - \text{mean}(\omega)| + \epsilon)$
- Transient cutoff: $t^* = 950.0$

### Clustering

- **Method**: k-NN (k=1)
- **Template ICs**:
  - FP: $[0.4, 0.0]$ — Fixed point (stable equilibrium)
  - LC: $[2.7, 0.0]$ — Limit cycle (rotational motion)

## Reproduction Code

### Setup

{{ load_snippet("case_studies/pendulum/setup_pendulum_system.py::setup_pendulum_system") }}

### Main Estimation

{{ load_snippet("case_studies/pendulum/main_pendulum_case1.py::main") }}

## Case 1: Baseline Results

Results below compare the CPU-forced regression run against the same pipeline run end-to-end
on GPU (sampling, integration via `JaxSolver`, and classification via the GPU-backed
`KNeighborsClassifier` with real cuML) -- see
[`tests/integration/pendulum/test_pendulum_gpu.py`](https://github.com/adrianwix/pybasin/blob/main/tests/integration/pendulum/test_pendulum_gpu.py).

{{ cpu_gpu_tabs("pendulum_case1", [
    ("Basin Stability", "pendulum_case1_basin_stability.png"),
    ("State Space", "pendulum_case1_state_space.png"),
    ("Feature Space", "pendulum_case1_feature_space.png"),
    ("Trajectories", "pendulum_case1_trajectories.png"),
]) }}

## Case 2: Parameter Sweep

{{ cpu_gpu_tabs("pendulum_case2", [
    ("Basin Stability Variation", "pendulum_case2_basin_stability_variation.png"),
    ("Bifurcation Diagram", "pendulum_case2_bifurcation_diagram.png"),
]) }}

## Case 3: Sample Size Convergence Study

This hyperparameter study varies the number of initial conditions $N$ from ~50 to ~5000 (using $5 \times \text{logspace}(1, 3, 20)$) to assess how basin stability estimates converge as sample size increases. The relative standard error decreases as $\text{SE}/\mathcal{S}_{\mathcal{B}} \sim 1/\sqrt{N}$.

{{ cpu_gpu_tabs("pendulum_case3", [
    ("Basin Stability Variation", "pendulum_case3_basin_stability_variation.png"),
]) }}
