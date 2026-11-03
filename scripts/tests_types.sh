#!/usr/bin/env bash

# the MYPY cache is currently broken and we thus clear it every time
rm -rf ../.mypy_cache

./run_tests.py --types