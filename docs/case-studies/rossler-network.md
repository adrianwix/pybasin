# Rössler Network

## System Description

Network of 100 coupled Rössler oscillators studying synchronization dynamics:

$$
\begin{aligned}
\dot{x}_i &= -y_i - z_i + K \sum_{j \in \mathcal{N}_i} (x_j - x_i) \\
\dot{y}_i &= x_i + ay_i \\
\dot{z}_i &= b + z_i(x_i - c)
\end{aligned}
$$

where $i = 1, \ldots, 100$ and $\mathcal{N}_i$ denotes the neighbors of node $i$ in the network.

### System Parameters

| Parameter         | Symbol | Value                 |
| ----------------- | ------ | --------------------- |
| Rössler parameter | $a$    | 0.2                   |
| Rössler parameter | $b$    | 0.2                   |
| Rössler parameter | $c$    | 7.0                   |
| Coupling strength | $K$    | 0.218 (baseline)      |
| Network topology  | —      | Scale-free, 100 nodes |

### Sampling

- **Dimension**: $D = 300$ (3 states $\times$ 100 nodes)
- **Sample size**: $N = 500$
- **Distribution**: $\rho$ = Uniform
- **Region of interest**: $\mathcal{Q}(x_i, y_i, z_i) : [-15, 15] \times [-15, 15] \times [-5, 35]$ per node

### Solver

| Setting            | Value                                 |
| ------------------ | ------------------------------------- |
| Method             | Dopri5 (Diffrax)                      |
| Time span          | $[0, 1000]$                           |
| Steps              | 1000 ($f_s$ = 1 Hz)                   |
| Relative tolerance | 1e-03                                 |
| Absolute tolerance | 1e-06                                 |
| Event function     | Divergence at $\lvert y \rvert > 400$ |

### Feature Extraction

Maximum pairwise deviation at final time step:

- States: all $x_i, y_i, z_i$
- Formula: $\Delta_x = \max_i(x_i) - \min_i(x_i)$, similarly for $y$, $z$; plus $\Delta_{\text{all}} = \max(\Delta_x, \Delta_y, \Delta_z)$
- Transient cutoff: $t^* = 950.0$

### Clustering

- **Method**: Threshold classifier (`SynchronizationClassifier`)
- **Threshold**: $\epsilon = 1.5$
- **Rule**: Synchronized if $\Delta_{\text{all}} < \epsilon$, desynchronized otherwise

### Attractors

The system exhibits three types of behavior:

- **Synchronized**: All oscillators converge to a common trajectory
- **Desynchronized**: Oscillators remain coupled but do not synchronize
- **Unbounded**: Some trajectories diverge to infinity (detected by event function)

Basin stability is computed for non-unbounded states (synchronized + desynchronized).

## Reproduction Code

### Setup

{{ load_snippet("case_studies/rossler_network/setup_rossler_network_system.py::setup_rossler_network_system") }}

### Single K Value

{{ load_snippet("case_studies/rossler_network/main_rossler_network.py::main") }}

### K Parameter Sweep

{{ load_snippet("case_studies/rossler_network/main_rossler_network_k_study.py::main") }}

## Baseline Results (K=0.218)

Note that this case study never forces CPU in the first place (`setup_rossler_network_system`
already picks GPU whenever CUDA is available), so the GPU tab below is mainly explicit,
separately-tracked evidence rather than a device-behavior difference -- see
[`tests/integration/rossler_network/test_rossler_network_gpu.py`](https://github.com/adrianwix/pybasin/blob/main/tests/integration/rossler_network/test_rossler_network_gpu.py).

{{ cpu_gpu_tabs("rossler_network_baseline", [
    ("Basin Stability", "rossler_network_baseline_basin_stability.png"),
    ("State Space", "rossler_network_baseline_state_space.png"),
    ("Feature Space", "rossler_network_baseline_feature_space.png"),
]) }}

## K Parameter Sweep

{{ cpu_gpu_tabs("rossler_network_k_sweep", [
    ("Basin Stability Variation", "rossler_network_k_sweep_basin_stability_variation.png"),
]) }}

## References

Menck, P. J., Heitzig, J., Marwan, N., & Kurths, J. (2013). _How basin stability complements the linear-stability paradigm_. Nature Physics, 9(2), 89-92.
