#!/bin/bash

set -e

rm -rf dist/
source .env
uv build --package pybasin --out-dir dist
uv publish dist/*
