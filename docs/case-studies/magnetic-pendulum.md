# Magnetic Pendulum

A damped pendulum moving above three magnets has three attracting equilibria separated
by intricate basin boundaries. This case checks pybasin's histogram and pairwise
groupers against labels generated independently by Attractors.jl.

The model is a variant of the magnetic pendulum of Motter et al. (2013):

$$
\begin{aligned}
\ddot{x} &= -\omega^2 x - \alpha \dot{x} - \sum_{i=1}^{3} \frac{\gamma (x - x_i)}{D_i^3}, \\
\ddot{y} &= -\omega^2 y - \alpha \dot{y} - \sum_{i=1}^{3} \frac{\gamma (y - y_i)}{D_i^3}, \\
D_i &= \sqrt{(x - x_i)^2 + (y - y_i)^2 + d^2}.
\end{aligned}
$$

The magnets $(x_i, y_i)$ sit on the unit circle at angles $2\pi i/3$. Motter et al. place
them on a triangle of unit edge length instead. The parameters $\omega = 0.8$,
$\alpha = 0.2$, $d = 0.2$, $\gamma = 1$ are those of the Attractors.jl test suite.

Attractors.jl samples 1,000 initial positions from $[-2, 2]^2$, fixes both initial
velocities at zero, integrates to $t=500$ with `Vern9`, and groups the final positions.
Pybasin receives the exact same four-dimensional initial conditions through
`CsvSampler`. Its trajectory integration and feature extraction are independent of the
Julia implementation.

## Validation method

MCC and every per-class F1 score must be at least 0.98. The confusion matrix and paired
mismatch rate expose individual classification differences, while the basin table
reports both sampling-error definitions returned by the basin-stability estimator:
$e_{abs}=\sqrt{S_B(1-S_B)/N}$ and $e_{rel}=1/\sqrt{NS_B}$. Basin fractions must remain
within 1.96 combined standard errors of the reference fractions.

### Histogram grouping

{{ accuracy_cpu_gpu_tabs("magnetic_pendulum_histogram") }}

### Pairwise grouping

{{ accuracy_cpu_gpu_tabs("magnetic_pendulum_pairwise") }}

## Reproduction

```bash
julia --project=packages/pybasin/tests/accuracy/_julia_project \
    packages/pybasin/tests/accuracy/magnetic_pendulum/generate_ground_truth.jl

uv run pytest packages/pybasin/tests/accuracy/magnetic_pendulum \
    --generate-accuracy-artifacts
```

The CPU and GPU tests share one code path and one reference CSV. GPU evidence is
published only after execution on a CUDA device; a skipped test leaves no GPU record.

## References

Motter, A. E., Gruiz, M., Károlyi, G., & Tél, T. (2013). _Doubly transient chaos: Generic form of chaos in autonomous dissipative systems_. Physical Review Letters, 111(19), 194101. https://doi.org/10.1103/PhysRevLett.111.194101

Datseris, G., Rossi, K. L., & Wagemakers, A. (2023). _Framework for global stability analysis of dynamical systems_. Chaos, 33(7), 073151. https://doi.org/10.1063/5.0159675

Datseris, G., Rossi, K. L., Wagemakers, A., et al. (2026). _Attractors.jl_ (v1.38.4) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20729175
