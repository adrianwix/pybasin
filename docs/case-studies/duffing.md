# Duffing Oscillator

## System Description

Duffing oscillator with cubic nonlinearity:

$$
\begin{aligned}
\dot{x} &= v \\
\dot{v} &= -\delta v - k_3 x^3 + A \cos(t)
\end{aligned}
$$

### System Parameters

| Parameter           | Symbol   | Value |
| ------------------- | -------- | ----- |
| Damping coefficient | $\delta$ | 0.08  |
| Cubic stiffness     | $k_3$    | 1.0   |
| Forcing amplitude   | $A$      | 0.2   |

### Sampling

- **Dimension**: $D = 2$
- **Sample size**: $N = 10000$
- **Distribution**: $\rho$ = Uniform
- **Region of interest**: $\mathcal{Q}(x, v) : [-1, 1] \times [-0.5, 1]$

### Solver

| Setting            | Value               |
| ------------------ | ------------------- |
| Method             | Dopri5 (Diffrax)    |
| Time span          | $[0, 1000]$         |
| Steps              | 5000 ($f_s$ = 5 Hz) |
| Relative tolerance | 1e-08               |
| Absolute tolerance | 1e-06               |

### Feature Extraction

Maximum and standard deviation of position:

- States: $x$ (state 0)
- Formula: $[\max(x), \sigma(x)]$
- Transient cutoff: $t^* = 900.0$

### Clustering

- **Method**: k-NN (k=1)
- **Template ICs**:
  - y1: $[-0.21, 0.02]$ — Period-1 limit cycle (small amplitude)
  - y2: $[1.05, 0.77]$ — Period-1 limit cycle (large amplitude)
  - y3: $[-0.67, 0.02]$ — Period-2 limit cycle
  - y4: $[-0.46, 0.30]$ — Period-2 limit cycle (symmetric)
  - y5: $[-0.43, 0.12]$ — Period-3 limit cycle

## Reproduction Code

### Setup

{{ load_snippet("case_studies/duffing_oscillator/setup_duffing_oscillator_system.py::setup_duffing_oscillator_system") }}

### Main Estimation

{{ load_snippet("case_studies/duffing_oscillator/main_duffing_oscillator_supervised.py::main") }}

## Case 1: Baseline Results (Supervised)

Results below compare the CPU-forced regression run against the same pipeline run end-to-end
on GPU (sampling, integration via `JaxSolver`, and classification via the GPU-backed
`KNeighborsClassifier` with real cuML) -- see
[`tests/integration/duffing/test_duffing_gpu.py`](https://github.com/adrianwix/pybasin/blob/main/tests/integration/duffing/test_duffing_gpu.py).

{{ cpu_gpu_tabs("duffing_case1", [
    ("Basin Stability", "duffing_case1_basin_stability.png"),
    ("State Space", "duffing_case1_state_space.png"),
    ("Feature Space", "duffing_case1_feature_space.png"),
    ("Trajectories", "duffing_case1_trajectories.png"),
    ("Phase Space", "duffing_case1_phase_space.png"),
], gpu_images=[
    ("Basin Stability", "duffing_case1_gpu_basin_stability.png"),
    ("State Space", "duffing_case1_gpu_state_space.png"),
    ("Feature Space", "duffing_case1_gpu_feature_space.png"),
    ("Trajectories", "duffing_case1_gpu_trajectories.png"),
]) }}

## Case 2: Unsupervised Clustering with Template Relabeling

This case demonstrates unsupervised attractor discovery using DBSCAN clustering, followed by relabeling using KNN template matching to assign meaningful attractor names. The GPU tab runs the same DBSCAN + KNN-template relabeling pipeline end-to-end on GPU via cuML/`JaxSolver`.

{{ cpu_gpu_tabs("duffing_case2", [
    ("Basin Stability", "duffing_case2_basin_stability.png"),
    ("State Space", "duffing_case2_state_space.png"),
    ("Feature Space", "duffing_case2_feature_space.png"),
]) }}
