#!/usr/bin/env bash
set -e

echo "Running CI checks..."
echo ""

echo "==> Syncing the workspace..."
uv sync

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
echo "✓ All CI checks passed!"
