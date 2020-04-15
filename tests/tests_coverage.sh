#!/bin/bash
echo 'Determine coverage of all unittests...'

CORES=`python3 -c 'from multiprocessing import cpu_count; print(cpu_count() // 2)'`

export NUMBA_DISABLE_JIT=1
export MPLBACKEND="agg"

mkdir -p coverage
python3 -m pytest -c pytest.ini -n $CORES \
    --durations=10 \
	--cov-report html:coverage \
	--cov=pde ..
