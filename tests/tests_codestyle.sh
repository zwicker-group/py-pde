#!/usr/bin/env bash
#
# This script checks the code format of this package without changing files
#

for dir in pde examples ; do
    echo "Checking codestyle in folder ${dir}:"

    # format imports
    isort --profile black --diff ../${dir}

    # black format all code
    black -t py36 --check ../${dir}
done