#!/bin/bash

# Run all tests, even if they are marked as slow
echo 'Run all unittests:'
./run_tests.py --unit --runslow --num_cores auto -- -rsx
