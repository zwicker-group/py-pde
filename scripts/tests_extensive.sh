#!/bin/bash

# test pattern was not specified
echo 'Run all unittests:'
./run_tests.py --unit --runslow --num_cores auto
