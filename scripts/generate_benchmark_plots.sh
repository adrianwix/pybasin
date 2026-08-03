#!/usr/bin/env bash
# Generate benchmark comparison plots for documentation
# These plots use CPSME styling for thesis-quality output
#
# Usage:
#   bash scripts/generate_benchmark_plots.sh         # pybasin + bSTAB only
#   bash scripts/generate_benchmark_plots.sh --all    # all tools (+ Attractors.jl, pynamicalsys)
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

echo ""
echo "=== Generating End-to-End Benchmark Plots (All Tools) ==="
if [[ "${1:-}" == "--all" ]]; then
    uv run python -m benchmarks.end_to_end.compare_all --all
else
    uv run python -m benchmarks.end_to_end.compare_all
fi

echo ""
echo "=== Generating Solver Comparison Plots ==="
uv run python -m benchmarks.solver_comparison.compare_all

echo ""
echo "=== Done! ==="
echo "Plots generated in:"
echo "  - packages/pybasin/docs/assets/benchmarks/end_to_end/"
echo "  - packages/pybasin/docs/assets/benchmarks/solver_comparison/"
