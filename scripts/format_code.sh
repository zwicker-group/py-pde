#!/usr/bin/env bash
# This script formats the code of this package

echo "Upgrading python syntax..."
pushd .. > /dev/null
find . -name '*.py' -exec pyupgrade --py39-plus {} +
popd > /dev/null

echo "Formating import statements..."
ruff check --fix --config=../pyproject.toml ..

echo "Formating docstrings..."
docformatter --in-place --black --recursive ..

echo "Formating source code..."
ruff format --config=../pyproject.toml ..