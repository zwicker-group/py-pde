#!/usr/bin/env bash
# This script formats the code of this package

echo "Formating import statements..."
isort ..

echo "Formating source code..."
black ..