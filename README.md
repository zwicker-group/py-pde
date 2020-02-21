# py-pde

[![Build Status](https://travis-ci.org/zwicker-group/py-pde.svg?branch=master)](https://travis-ci.org/zwicker-group/py-pde)
[![codecov](https://codecov.io/gh/zwicker-group/py-pde/branch/master/graph/badge.svg)](https://codecov.io/gh/zwicker-group/py-pde)
[![PyPI version](https://badge.fury.io/py/py-pde.svg)](https://badge.fury.io/py/py-pde)
[![Documentation Status](https://readthedocs.org/projects/py-pde/badge/?version=latest)](https://py-pde.readthedocs.io/en/latest/?badge=latest)
      

Python package for solving partial differential equations (PDEs). 
This package provides classes for scalar and tensor fields discretized on grids
as well as associated differential operators.
This allows defining, inspecting, and solving typical PDEs that appear for
instance in the study of dynamical systems in physics.
A simple example showing the evolution of the diffusion equation in 2d is given
by

```python
from pde.common import *

grid = UnitGrid([64, 64])                 # generate grid
state = ScalarField.random_uniform(grid)  # generate initial condition

eq = DiffusionPDE(diffusivity=0.1)        # define the pde
result = eq.solve(state, t_range=10)      # solve the pde
result.plot()                             # plot the resulting field
```

More examples illustrating the capabilities of the package can be found in the
folder `examples`.
A detailed [documentation is available on readthedocs](https://py-pde.readthedocs.io/)
and as [a single PDF file](https://py-pde.readthedocs.io/_/downloads/en/latest/pdf/).

