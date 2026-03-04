#!/usr/bin/env bash
set -e

echo "Running CI checks..."
echo ""

echo "==> Syncing all dependencies (including optional extras)..."
uv sync --all-extras --all-groups

echo ""
echo "==> Running ruff linter..."
uv run ruff check --fix --output-format=concise

echo ""
echo "==> Running ruff formatter..."
uv run ruff format

echo ""
echo "==> Running pyright type checker..."
uv run pyright

echo ""
echo "==> Running pyright for plotters (basic checking)..."
uv run pyright -p src/pybasin/plotters/pyrightconfig.json

echo ""
echo "==> Running pyright for benchmarks (basic checking)..."
uv run pyright -p benchmarks/pyrightconfig.json

echo ""
echo "✓ All CI checks passed!"
