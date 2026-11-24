# Hénon Map

A small, fast, genuinely chaotic-or-divergent discrete map, used to validate
`HistogramClusterer` and `PairwiseComparisonClusterer` against real `Attractors.jl`
output. See `henon_map.py`/`henon_map_ode.py` for the map itself and
`henon_feature_extractor.py` for the mean-state-with-sentinel featurizer.

## Run the experiments (Python)

```bash
uv run python -m case_studies.henon_map.main_henon_histogram_experiment
uv run python -m case_studies.henon_map.main_henon_pairwise_experiment
```

Each samples initial conditions, iterates the map, extracts features, groups them
with the respective clusterer, and opens a `MatplotlibPlotter` dashboard (basin
stability bars, state space, feature space).

## Generate ground truth / run the benchmark (Julia)

Reproduces the real `Attractors.jl` `GroupViaHistogram` pipeline this case study is
validated against (see `packages/pybasin/tests/accuracy/henon_map/test_henon_map.py`),
and times pybasin against it:

```bash
julia --project=packages/pybasin/tests/accuracy/_julia_project \
    packages/pybasin/tests/accuracy/henon_map/generate_ground_truth.jl

packages/pybasin/scripts/run_grouper_benchmarks.sh
```

Ground truth is written to `packages/pybasin/tests/accuracy/henon_map/ground_truths/
henon_map.csv`; benchmark results to `packages/pybasin/benchmarks/histogram/results/`, and
the comparison plot to `packages/pybasin/docs/assets/benchmarks/histogram/`. The runner
script also covers the magnetic pendulum's pairwise benchmark -- see that case study's
own `README.md` if you only want one grouper.
