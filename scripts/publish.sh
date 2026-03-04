#!/bin/bash

set -e # makes the script exit immediately if any command returns a non-zero exit code
rm -rf dist/ # makes sure old versions do not get published
source .env && uv build && uv publish
