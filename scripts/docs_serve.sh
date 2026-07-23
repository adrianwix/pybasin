#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

DOCS_HOST="${DOCS_HOST:-127.0.0.1}"
DOCS_PORT="${DOCS_PORT:-8000}"

exec uv run --no-sync mkdocs serve \
  --dev-addr "${DOCS_HOST}:${DOCS_PORT}" \
  --watch mkdocs.yml \
  --watch docs \
  --watch src \
  --watch packages/tstorch/src \
  --livereload \
  "$@"
