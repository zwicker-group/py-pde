# This script is used to compile the package and upload it to pypi

# change to base folder
cd ..

# build the distribution
python3 setup.py sdist bdist_wheel

# upload the package to pypi
# python3 -m twine upload dist/*

# clean temporary data 
rm -rf build dist py_pde.egg-info
