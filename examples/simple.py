#!/usr/bin/env python3

from pde.common import *

eq = DiffusionPDE(diffusivity=0.1)                  # define the pde
grid = UnitGrid([64, 64])                           # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

result = eq.solve(state, t_range=10)
result.plot(show=True)
