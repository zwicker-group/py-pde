from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'py-pde',
  package_data={"pde": ["py.typed"]},
  packages = find_packages(),
  zip_safe=False,  # this is required for mypy to find the py.typed file
  version = '0.8.0',
  license='MIT',
  description = 'Python package for solving partial differential equations',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'David Zwicker',
  author_email = 'david.zwicker@ds.mpg.de',
  url = 'https://github.com/zwicker-group/py-pde',
  download_url = 'https://github.com/zwicker-group/py-pde/archive/v0.8.0.tar.gz',
  keywords = ['pdes', 'partial-differential-equations', 'dynamical-systems'],
  python_requires='>=3.6',
  install_requires=['matplotlib',
                    'numpy',
                    'numba',
                    'scipy',
                    'sympy'],
  extras_require={
        "hdf":  ["h5py>=2"],
        "progress": ["tqdm>=4.40"],
        "interactive": ["napari>=0.3"]
    },
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)