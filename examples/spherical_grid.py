#!/usr/bin/env python3

from pde.grids import SphericalGrid
from pde.fields import ScalarField
from pde.pdes import DiffusionPDE

eq = DiffusionPDE()                                 # define the pde
grid = SphericalGrid(radius=5, shape=16)            # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

result = eq.solve(state, t_range=1, dt=0.005)
result.plot(show=True)
