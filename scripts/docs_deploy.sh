#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

uv run mkdocs build --strict --clean --config-file packages/pybasin/mkdocs.yml
uv run mkdocs build --strict --clean --config-file packages/tstorch/mkdocs.yml
uv run mkdocs build --strict --clean

mkdir -p site/pybasin site/tstorch
cp -r site-pybasin/. site/pybasin/
cp -r site-tstorch/. site/tstorch/

uv run mkdocs gh-deploy --force --dirty
