#!/bin/bash

echo 'Run serial tests to determine coverage...'
./run_tests.py --unit --coverage --nojit --num_cores auto --runslow

echo 'Run parallel tests to determine coverage...'
./run_tests.py --unit --coverage --nojit --use_mpi --runslow
