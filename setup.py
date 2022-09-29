# THIS FILE IS CREATED AUTOMATICALLY AND ALL MANUAL CHANGES WILL BE OVERWRITTEN
# If you want to adjust settings in this file, change scripts/templates/setup.py

from pathlib import Path

from setuptools import find_packages, setup

BASE_PATH = Path(__file__).resolve().parent

# read the version from the particular file
with open(BASE_PATH / "pde" / "version.py", "r") as f:
    exec(f.read())

DOWNLOAD_URL = f"https://github.com/zwicker-group/py-pde/archive/v{__version__}.tar.gz"

# read the description from the README file
with open(BASE_PATH / "README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="py-pde",
    packages=find_packages(),
    zip_safe=False,  # this is required for mypy to find the py.typed file
    version=__version__,
    license="MIT",
    description="Python package for solving partial differential equations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="David Zwicker",
    author_email="david.zwicker@ds.mpg.de",
    url="https://github.com/zwicker-group/py-pde",
    download_url=DOWNLOAD_URL,
    keywords=["pdes", "partial-differential-equations", "dynamical-systems"],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.1.0",
        "numba>=0.56.0",
        "numpy>=1.22.0",
        "scipy>=1.4.0",
        "sympy>=1.5.0",
        "tqdm>=4.60",
    ],
    extras_require={
        "hdf": ["h5py>=2.10"],
        "io": ["h5py>=2.10", "pandas>=1.2"],
        "interactive": ["napari>=0.4.8", "ipywidgets>=7"],
        "mpi": ["numba-mpi>=0.13"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
