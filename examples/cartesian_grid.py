#!/usr/bin/env python3

from pde.common import *

grid = CartesianGrid([[0, 1], [0, 2]], [3, 8])     # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3) # generate initial condition

eq = DiffusionPDE()                               # define the pde
result = eq.solve(state, t_range=10, dt=0.005)    # solve it
result.plot(show=True)
