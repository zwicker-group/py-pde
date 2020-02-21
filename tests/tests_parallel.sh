#!/bin/bash

CORES=`python3 -c 'from multiprocessing import cpu_count; print(cpu_count() // 2)'`

export NUMBA_WARNINGS=1
export MPLBACKEND="agg"

if [ ! -z $1 ] 
then 
	# test pattern was specified 
	echo 'Run unittests with pattern '$1' on '$CORES' cores:'
	python3 -m pytest -n $CORES --durations=10 -k "$1" . ..
else
	# test pattern was not specified
	echo 'Run all unittests on '$CORES' cores:'
	python3 -m pytest -n $CORES --durations=10 . ..
fi
