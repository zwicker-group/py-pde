#!/bin/bash
echo 'Determine coverage of all unittests...'

./run_tests.py --unit --coverage --no_numba --parallel --runslow
