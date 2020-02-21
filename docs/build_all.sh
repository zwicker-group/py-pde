#!/bin/sh

cd source
./run_autodoc.py
cd ..

make html
make latexpdf
make linkcheck