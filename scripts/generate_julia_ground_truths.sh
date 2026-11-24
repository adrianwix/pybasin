#!/usr/bin/env bash
set -e

echo "Generating Julia ground truths for HistogramClusterer / PairwiseComparisonClusterer..."
echo ""

JULIA_PROJECT="packages/pybasin/tests/accuracy/_julia_project"

echo "==> Full pipeline: Hénon map (histogram + pairwise)..."
julia --project="$JULIA_PROJECT" \
    packages/pybasin/tests/accuracy/henon_map/generate_ground_truth.jl

echo ""
echo "==> Full pipeline: magnetic pendulum (histogram + pairwise)..."
julia --project="$JULIA_PROJECT" \
    packages/pybasin/tests/accuracy/magnetic_pendulum/generate_ground_truth.jl

echo ""
echo "==> Algorithmic edge cases (dumb map, aggregation threshold, order permutation)..."
julia --project="$JULIA_PROJECT" \
    packages/pybasin/tests/accuracy/pairwise/generate_algorithmic_fixtures.jl

echo ""
echo "✓ Ground truths written to packages/pybasin/tests/accuracy/{henon_map,magnetic_pendulum}/ground_truths/"
echo "✓ Algorithmic fixtures written to packages/pybasin/tests/accuracy/pairwise/fixtures/"
