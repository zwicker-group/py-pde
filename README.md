# py-pde

[![PyPI version](https://badge.fury.io/py/py-pde.svg)](https://badge.fury.io/py/py-pde)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/py-pde.svg)](https://anaconda.org/conda-forge/py-pde)
[![Documentation Status](https://readthedocs.org/projects/py-pde/badge/?version=latest)](https://py-pde.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02158/status.svg)](https://doi.org/10.21105/joss.02158)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zwicker-group/py-pde/master?filepath=examples%2Fjupyter)

[![Build Status](https://travis-ci.org/zwicker-group/py-pde.svg?branch=master)](https://travis-ci.org/zwicker-group/py-pde)
[![codecov](https://codecov.io/gh/zwicker-group/py-pde/branch/master/graph/badge.svg)](https://codecov.io/gh/zwicker-group/py-pde)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/zwicker-group/py-pde.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/zwicker-group/py-pde/context:python)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

`py-pde` is a Python package for solving partial differential equations (PDEs). 
The package provides classes for grids on which scalar and tensor fields can be
defined. The associated differential operators are computed using a
numba-compiled implementation of finite differences. This allows defining,
inspecting, and solving typical PDEs that appear for instance in the study of
dynamical systems in physics. The focus of the package lies on easy usage to
explore the behavior of PDEs. However, core computations can be compiled
transparently using numba for speed.

[Try it out online!](https://mybinder.org/v2/gh/zwicker-group/py-pde/master?filepath=examples%2Fjupyter)


Installation
------------

`py-pde` is available on `pypi`, so you should be able to install it through
`pip`:

```bash
pip install py-pde
```

In order to have all features of the package available, you might also want to 
install the following optional packages:

```bash
pip install h5py pandas tqdm
```

Moreover, `ffmpeg` needs to be installed for creating movies.

As an alternative, you can install `py-pde` through [conda](https://docs.conda.io/en/latest/)
using [conda-forge](https://conda-forge.org/) channel:

```bash
conda install -c conda-forge py-pde
```

Installation with `conda` includes all required dependencies to have all features of `py-pde`.

Usage
-----

A simple example showing the evolution of the diffusion equation in 2d:

```python
import pde

grid = pde.UnitGrid([64, 64])                 # generate grid
state = pde.ScalarField.random_uniform(grid)  # generate initial condition

eq = pde.DiffusionPDE(diffusivity=0.1)        # define the pde
result = eq.solve(state, t_range=10)          # solve the pde
result.plot()                                 # plot the resulting field
```

PDEs can also be specified by simply writing expressions of the evolution rate.
For instance, the
[Cahn-Hilliard equation](https://en.wikipedia.org/wiki/Cahn–Hilliard_equation)
can be implemented as
```python
eq = pde.PDE({'c': 'laplace(c**3 - c - laplace(c))'})
```
which can be used in place of the `DiffusionPDE` in the example above.


More information
----------------
* Tutorial notebooks in the [tutorials folder](https://github.com/zwicker-group/py-pde/tree/master/examples/tutorial)
* [Examples gallery](https://py-pde.readthedocs.io/en/latest/examples_gallery/)
  with an overview of the capabilities of the package
* The [Discussions on GitHub](https://github.com/zwicker-group/py-pde/discussions)
* [Full documentation on readthedocs](https://py-pde.readthedocs.io/)
  or as [a single PDF file](https://py-pde.readthedocs.io/_/downloads/en/latest/pdf/).
* The [paper published in the Journal of Open Source Software](https://doi.org/10.21105/joss.02158)
 
