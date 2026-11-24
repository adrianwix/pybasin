# Magnetic Pendulum

A damped pendulum bob suspended above 3 point magnets, used to validate
`HistogramClusterer` and `PairwiseComparisonClusterer` against real `Attractors.jl`
output. See `magnetic_pendulum.py`/`magnetic_pendulum_ode.py` for the system itself
and `magnetic_pendulum_feature_extractor.py` for the final-position featurizer.
`setup_magnetic_pendulum_system.py` also fits a KNN classifier on Attractors.jl's own
named magnet templates (`magnet_1`/`magnet_2`/`magnet_3`).

## Run the experiments (Python)

```bash
uv run python -m case_studies.magnetic_pendulum.main_magnetic_pendulum_histogram_experiment
uv run python -m case_studies.magnetic_pendulum.main_magnetic_pendulum_pairwise_experiment
```

Each samples initial conditions, integrates the ODE, extracts the final position,
groups it with the respective clusterer, and opens a `MatplotlibPlotter` dashboard
(basin stability bars, state space, feature space).

## Generate ground truth / run the benchmark (Julia)

Reproduces the real `Attractors.jl` `GroupViaPairwiseComparison` pipeline this case
study is validated against (see
`packages/pybasin/tests/accuracy/magnetic_pendulum/test_magnetic_pendulum.py`), and
times pybasin against it:

```bash
julia --project=packages/pybasin/tests/accuracy/_julia_project \
    packages/pybasin/tests/accuracy/magnetic_pendulum/generate_ground_truth.jl

packages/pybasin/scripts/run_grouper_benchmarks.sh
```

Ground truth is written to `packages/pybasin/tests/accuracy/magnetic_pendulum/
ground_truths/magnetic_pendulum.csv`; benchmark results to `packages/pybasin/benchmarks/
pairwise/results/`, and the comparison plot to
`packages/pybasin/docs/assets/benchmarks/pairwise/`. The runner script also covers the
Hénon map's histogram benchmark -- see that case study's own `README.md` if you only
want one grouper.
