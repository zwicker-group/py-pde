#!/usr/bin/env bash
# This script formats the code of this package

echo "Upgrading python syntax..."
pushd .. > /dev/null
find . -name '*.py' -exec pyupgrade --py39-plus {} +
popd > /dev/null

echo "Formatting import statements..."
ruff check --fix --config=../pyproject.toml ..

echo "Formatting docstrings..."
docformatter --in-place --black --recursive ..

echo "Formatting source code..."
ruff format --config=../pyproject.toml ..