#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

DOCS_HOST="${DOCS_HOST:-127.0.0.1}"
DOCS_PORT="${DOCS_PORT:-8000}"
DOCS_SITE="${1:-all}"

case "${DOCS_SITE}" in
  root)
    config="mkdocs.yml"
    ;;
  pybasin)
    config="packages/pybasin/mkdocs.yml"
    ;;
  tstorch)
    config="packages/tstorch/mkdocs.yml"
    ;;
  all)
    export TORRSTACK_HOME_URL="http://${DOCS_HOST}:${DOCS_PORT}/"
    export PYBASIN_DOCS_URL="http://${DOCS_HOST}:${DOCS_PORT}/pybasin/"
    export TSTORCH_DOCS_URL="http://${DOCS_HOST}:${DOCS_PORT}/tstorch/"

    uv run --no-sync mkdocs build --clean --config-file packages/pybasin/mkdocs.yml
    uv run --no-sync mkdocs build --clean --config-file packages/tstorch/mkdocs.yml
    uv run --no-sync mkdocs build --clean

    rm -rf site/pybasin site/tstorch
    mkdir -p site/pybasin site/tstorch
    cp -r site-pybasin/. site/pybasin/
    cp -r site-tstorch/. site/tstorch/

    exec python3 -m http.server "${DOCS_PORT}" --bind "${DOCS_HOST}" --directory site
    ;;
  *)
    echo "usage: $0 [root|pybasin|tstorch|all] [mkdocs options...]" >&2
    exit 2
    ;;
esac

shift || true

exec uv run --no-sync mkdocs serve \
  --config-file "${config}" \
  --dev-addr "${DOCS_HOST}:${DOCS_PORT}" \
  --watch mkdocs.base.yml \
  --watch docs \
  --watch docs_tools \
  --watch packages/pybasin/docs \
  --watch packages/pybasin/src \
  --watch packages/tstorch/docs \
  --watch packages/tstorch/src \
  --livereload \
  "$@"
