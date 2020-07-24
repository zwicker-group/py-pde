from pathlib import Path
from setuptools import setup, find_packages


BASE_PATH = Path(__file__).resolve().parent


# read the version from the particular file
with open(BASE_PATH / "pde" / "version.py", "r") as f:
    exec(f.read())

DOWNLOAD_URL = f"https://github.com/zwicker-group/py-pde/archive/v{__version__}.tar.gz"


# read the version from the particular file
with open(BASE_PATH / "README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="py-pde",
    package_data={"pde": ["py.typed"]},
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
    python_requires=">=3.6",
    install_requires=["matplotlib", "numpy", "numba", "scipy", "sympy"],
    extras_require={
        "hdf": ["h5py>=2"],
        "progress": ["tqdm>=4.40"],
        "interactive": ["napari>=0.3"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
