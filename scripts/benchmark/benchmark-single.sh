#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULT_DIR="${PROJECT_ROOT}/benchmarks/tstorch/results"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
JSON_OUTPUT="${JSON_OUTPUT:-${RESULT_DIR}/single-feature-${TIMESTAMP}.json}"

mkdir -p "${RESULT_DIR}"
cd "${PROJECT_ROOT}"

uv run pytest packages/tstorch/tests/benchmarks \
  -x \
  -k "single_feature" \
  --benchmark-json="${JSON_OUTPUT}"

echo "Benchmark JSON saved to ${JSON_OUTPUT}"
