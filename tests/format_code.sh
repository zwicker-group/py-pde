#!/usr/bin/env bash
#
# This script formats the code of this package
#

for dir in pde examples ; do
    echo "Formating files in ${dir}:"

    # format imports
    isort --profile black ../${dir}

    # black format all code
    black -t py36 ../${dir}
done