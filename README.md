# py-pde

[![Build Status](https://travis-ci.org/zwicker-group/py-pde.svg?branch=master)](https://travis-ci.org/zwicker-group/py-pde)
[![codecov](https://codecov.io/gh/zwicker-group/py-pde/branch/master/graph/badge.svg)](https://codecov.io/gh/zwicker-group/py-pde)
[![PyPI version](https://badge.fury.io/py/py-pde.svg)](https://badge.fury.io/py/py-pde)
[![Documentation Status](https://readthedocs.org/projects/py-pde/badge/?version=latest)](https://py-pde.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zwicker-group/py-pde/master?filepath=examples%2Fjupyter)

`py-pde` is a Python package for solving partial differential equations (PDEs). 
The package provides classes for scalar and tensor fields discretized on grids
as well as associated differential operators.
This allows defining, inspecting, and solving typical PDEs that appear for
instance in the study of dynamical systems in physics.
The focus of the package lies on easy usage to explore the behavior of PDEs.
However, core computations can be compiled transparently using numba for speed.

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

Moreover, `ffmpeg` needs to be installed and for creating movies.


Usage
-----

A simple example showing the evolution of the diffusion equation in 2d:

```python
from pde.common import *

grid = UnitGrid([64, 64])                 # generate grid
state = ScalarField.random_uniform(grid)  # generate initial condition

eq = DiffusionPDE(diffusivity=0.1)        # define the pde
result = eq.solve(state, t_range=10)      # solve the pde
result.plot()                             # plot the resulting field
```

More examples illustrating the capabilities of the package can be found in the
 `examples` folder.
A detailed [documentation is available on readthedocs](https://py-pde.readthedocs.io/)
and as [a single PDF file](https://py-pde.readthedocs.io/_/downloads/en/latest/pdf/).

