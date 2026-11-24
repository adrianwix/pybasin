#!/usr/bin/env bash
# Run the histogram/pairwise grouper benchmarks (pybasin CPU/CUDA + Attractors.jl) and
# generate their comparison reports. Package-specific to pybasin's predictors module,
# unlike the multi-package benchmarks under the repo-root benchmarks/ directory (see
# AGENTS.md's "Benchmarks" section).
set -e

cd "$(git rev-parse --show-toplevel)"

JULIA_PROJECT="packages/pybasin/tests/accuracy/_julia_project"

echo "==> Histogram grouper: Attractors.jl"
julia --project="$JULIA_PROJECT" packages/pybasin/benchmarks/histogram/benchmark_julia.jl

echo "==> Histogram grouper: pybasin"
uv run python -m packages.pybasin.benchmarks.histogram.benchmark_pybasin

echo "==> Pairwise grouper: Attractors.jl"
julia --project="$JULIA_PROJECT" packages/pybasin/benchmarks/pairwise/benchmark_julia.jl

echo "==> Pairwise grouper: pybasin"
uv run python -m packages.pybasin.benchmarks.pairwise.benchmark_pybasin

echo "==> Comparison reports"
uv run python -m packages.pybasin.benchmarks.histogram.compare
uv run python -m packages.pybasin.benchmarks.pairwise.compare

echo ""
echo "Done. Plots written to packages/pybasin/docs/assets/benchmarks/{histogram,pairwise}/"
