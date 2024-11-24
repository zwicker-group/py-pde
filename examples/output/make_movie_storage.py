"""
Create a movie from a storage
=============================

This example shows how to create a movie from data stored during a simulation. Making
movies requires that `ffmpeg` is installed in a standard location.
"""

from pde import DiffusionPDE, MemoryStorage, ScalarField, UnitGrid, movie_scalar

grid = UnitGrid([16, 16])  # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

storage = MemoryStorage()  # create storage
tracker = storage.tracker(interrupts=1)  # create associated tracker

eq = DiffusionPDE()  # define the physics
eq.solve(state, t_range=2, dt=0.005, tracker=tracker)

# create movie from stored data
movie_scalar(storage, "diffusion.mov")
