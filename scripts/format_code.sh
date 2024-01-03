#!/usr/bin/env bash
# This script formats the code of this package

echo "Upgrading python syntax..."
pyupgrade --py37-plus ../**/*.py

echo "Formating import statements..."
isort ..

echo "Formating source code..."
black ..