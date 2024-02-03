#!/bin/bash

export PYTHONPATH=../py-pde  # likely path of pde package, relative to current base path

if [ ! -z $1 ] 
then 
    # test pattern was specified 
    echo 'Run unittests with pattern '$1':'
    ./run_tests.py --unit --runslow --nojit --pattern "$1"
else
    # test pattern was not specified
    echo 'Run all unittests:'
    ./run_tests.py --unit --nojit
fi
