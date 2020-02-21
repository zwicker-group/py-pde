#!/usr/bin/env python3

from pde.common import *

eq = DiffusionPDE()                                 # define the physics
grid = UnitGrid([16, 16])                           # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

storage = MemoryStorage()               # create storage
tracker = storage.tracker(interval=1)   # create associated tracker

eq.solve(state, t_range=2, dt=0.005, tracker=tracker)

# create movie from stored data
movie_scalar(storage, '/tmp/diffusion.mov')
