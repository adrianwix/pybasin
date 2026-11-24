# Hénon Map

The Hénon map provides a solver-independent check of pybasin's histogram and pairwise
groupers. Attractors.jl generates the initial conditions, iterates the map, extracts the
mean-state features, and assigns the reference labels. Pybasin reads those same initial
conditions through `CsvSampler`.

$$
\begin{aligned}
x_{n+1} &= 1 - 1.4x_n^2 + y_n, \\
y_{n+1} &= 0.3x_n.
\end{aligned}
$$

The reference dataset contains 1,000 initial conditions sampled from
$[-2, 2] \times [-2, 2]$. Both implementations iterate each point for 200 steps. The
Attractors.jl generator is pinned by the Julia manifest and records labels from
`GroupViaHistogram` and `GroupViaPairwiseComparison` in separate CSV columns.

## Validation method

Each CPU or GPU run uses the same CSV rows. MCC is the primary agreement criterion,
with a minimum of 0.99. Every per-class F1 score must also reach 0.99. The tables report
the confusion-derived mismatch count and the formal absolute and relative sampling
errors for each basin. Aggregate basin fractions must differ by no more than 1.96
combined standard errors.

### Histogram grouping

{{ accuracy_cpu_gpu_tabs("henon_map_histogram") }}

### Pairwise grouping

{{ accuracy_cpu_gpu_tabs("henon_map_pairwise") }}

## Reproduction

Generate the independent labels, then run the accuracy suite and refresh its committed
documentation records:

```bash
julia --project=packages/pybasin/tests/accuracy/_julia_project \
    packages/pybasin/tests/accuracy/henon_map/generate_ground_truth.jl

uv run pytest packages/pybasin/tests/accuracy/henon_map \
    --generate-accuracy-artifacts
```

GPU records are produced only when the test runs on a CUDA host. A skipped GPU test does
not constitute GPU validation.

## References

Hénon, M. (1976). _A two-dimensional mapping with a strange attractor_. Communications in Mathematical Physics, 50(1), 69-77. https://doi.org/10.1007/BF01608556

Datseris, G., Rossi, K. L., & Wagemakers, A. (2023). _Framework for global stability analysis of dynamical systems_. Chaos, 33(7), 073151. https://doi.org/10.1063/5.0159675

Datseris, G., Rossi, K. L., Wagemakers, A., et al. (2026). _Attractors.jl_ (v1.38.4) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20729175
