#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/benchmark/benchmark-feature.sh --match <domain>[/<feature>]
       scripts/benchmark/benchmark-feature.sh --match <domain>[/<feature>] --refresh-tsfresh

Runs the numerical-accuracy unit tests, their torch.compile(fullgraph=True) safety checks, and
the performance benchmarks defined for the given calculator domain (test_<domain>.py under
packages/tstorch/tests/{unit,benchmarks/calculators}/), or one feature within it, then:
  1. upserts all four results into packages/tstorch/reports/store.json, keyed by feature (and,
     within a feature, by variant/backend for performance or by param combo for precision/
     compile/export) - other features already recorded are untouched.
  2. regenerates packages/tstorch/reports/report.md from the full store.

Cached tsfresh benchmark measurements are reused when the stored fingerprint still matches the
tsfresh version, feature, params, benchmark variant, and input tensor. Pass --refresh-tsfresh to
remeasure them anyway.

Examples:
  scripts/benchmark/benchmark-feature.sh --match autocorrelation
  scripts/benchmark/benchmark-feature.sh --match autocorrelation/agg_autocorrelation
EOF
}

MATCH=""
REFRESH_TSFRESH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --match)
      MATCH="$2"
      shift 2
      ;;
    --refresh-tsfresh)
      REFRESH_TSFRESH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${MATCH}" ]]; then
  echo "error: --match <domain>[/<feature>] is required" >&2
  usage
  exit 1
fi

DOMAIN="${MATCH%%/*}"
FEATURE=""
if [[ "${MATCH}" == */* ]]; then
  FEATURE="${MATCH#*/}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UNIT_TEST_FILE="${PROJECT_ROOT}/packages/tstorch/tests/unit/test_${DOMAIN}.py"
BENCH_TEST_FILE="${PROJECT_ROOT}/packages/tstorch/tests/benchmarks/calculators/test_${DOMAIN}.py"
REPORT_DIR="${PROJECT_ROOT}/packages/tstorch/reports"
RAW_DIR="${REPORT_DIR}/raw"
STORE="${REPORT_DIR}/store.json"
REPORT="${REPORT_DIR}/report.md"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SLUG="${DOMAIN}${FEATURE:+-${FEATURE}}-${TIMESTAMP}"
PRECISION_JSON="${RAW_DIR}/precision-${SLUG}.json"
COMPILE_JSON="${RAW_DIR}/compile-${SLUG}.json"
EXPORT_JSON="${RAW_DIR}/export-${SLUG}.json"
BENCH_JSON="${RAW_DIR}/benchmark-${SLUG}.json"

if [[ ! -f "${BENCH_TEST_FILE}" ]]; then
  echo "error: no benchmark file for domain '${DOMAIN}' (expected ${BENCH_TEST_FILE})" >&2
  exit 1
fi

mkdir -p "${RAW_DIR}"
cd "${PROJECT_ROOT}"

if [[ -f "${UNIT_TEST_FILE}" ]]; then
  UNIT_ARGS=(
    "${UNIT_TEST_FILE}"
    "--precision-json=${PRECISION_JSON}"
    "--compile-json=${COMPILE_JSON}"
    "--export-json=${EXPORT_JSON}"
  )
  if [[ -n "${FEATURE}" ]]; then
    UNIT_ARGS+=(-k "${FEATURE}")
  fi
  uv run pytest "${UNIT_ARGS[@]}"
  uv run python "${SCRIPT_DIR}/tstorch_bench_store.py" upsert-precision "${PRECISION_JSON}" --store "${STORE}"
  uv run python "${SCRIPT_DIR}/tstorch_bench_store.py" upsert-compile "${COMPILE_JSON}" --store "${STORE}"
  uv run python "${SCRIPT_DIR}/tstorch_bench_store.py" upsert-export "${EXPORT_JSON}" --store "${STORE}"
else
  echo "note: no unit test file for domain '${DOMAIN}' (expected ${UNIT_TEST_FILE}) - skipping precision/compile/export" >&2
fi

BENCH_ARGS=("${BENCH_TEST_FILE}" "--benchmark-json=${BENCH_JSON}")
if [[ -n "${FEATURE}" ]]; then
  BENCH_ARGS+=(-k "${FEATURE}")
fi
if [[ "${REFRESH_TSFRESH}" == "1" ]]; then
  TSTORCH_BENCHMARK_STORE="${STORE}" TSTORCH_BENCHMARK_REMEASURE_TSFRESH=1 uv run pytest "${BENCH_ARGS[@]}"
else
  TSTORCH_BENCHMARK_STORE="${STORE}" uv run pytest "${BENCH_ARGS[@]}"
fi
uv run python "${SCRIPT_DIR}/tstorch_bench_store.py" upsert-performance "${BENCH_JSON}" --store "${STORE}"

uv run python "${SCRIPT_DIR}/tstorch_bench_store.py" render --store "${STORE}" --report "${REPORT}"

echo ""
echo "Store:  ${STORE}"
echo "Report: ${REPORT}"
