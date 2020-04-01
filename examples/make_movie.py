"""
Create a movie
==============

This example shows how to create a movie, which is only possible if `ffmpeg` is
installed in a standard location.
"""

from pde import UnitGrid, ScalarField, DiffusionPDE, MemoryStorage, movie_scalar

grid = UnitGrid([16, 16])                           # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

storage = MemoryStorage()               # create storage
tracker = storage.tracker(interval=1)   # create associated tracker

eq = DiffusionPDE()                                 # define the physics
eq.solve(state, t_range=2, dt=0.005, tracker=tracker)

# create movie from stored data
movie_scalar(storage, '/tmp/diffusion.mov')
