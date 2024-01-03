#!/usr/bin/env bash
# This script formats the code of this package

echo "Upgrading python syntax..."
pushd .. > /dev/null
pyupgrade --py38-plus **/*.py
popd > /dev/null

echo "Formating import statements..."
isort ..

echo "Formating source code..."
black ..