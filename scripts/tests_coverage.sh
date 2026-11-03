#!/bin/bash

export PYPDE_TESTRUN="1"
export NUMBA_DISABLE_JIT="1"
export JAX_DISABLE_JIT="1"
export MPLBACKEND="agg"

pushd ..

echo 'Run serial tests to determine coverage...'
export COVERAGE_FILE=".coverage.serial"
pytest --cov-config=pyproject.toml --cov=pde -n auto tests 

echo 'Run parallel tests to determine coverage...'
mpiexec -n 2 bash -lc '
    export COVERAGE_FILE=".coverage.mpi.${OMPI_COMM_WORLD_RANK:-0}"
    export TMPDIR="$(pwd)/.tmp_mpi_${OMPI_COMM_WORLD_RANK:-0}"
    mkdir -p "$TMPDIR"
    pytest --cov-config=pyproject.toml --cov=pde -p no:cacheprovider --use_mpi tests
    rm -rf "$TMPDIR"
'

# combine per-rank coverage data into the main file
coverage combine --keep --data-file=.coverage-total .coverage.serial .coverage.mpi.*
# create coverage report
coverage html -d scripts/coverage --data-file=.coverage-total
# delete temporary files
rm .coverage*

popd