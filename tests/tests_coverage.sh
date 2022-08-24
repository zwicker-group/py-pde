#!/bin/bash
echo 'Determine coverage of all unittests...'

./run_tests.py --unit --coverage --nojit --num_cores auto --runslow
