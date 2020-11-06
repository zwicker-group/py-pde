"""
Create a movie live
===================

This example shows how to create a movie while running the simulation. Making movies
requires that `ffmpeg` is installed in a standard location.
"""

from pde import DiffusionPDE, PlotTracker, ScalarField, UnitGrid

grid = UnitGrid([16, 16])  # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

tracker = PlotTracker(movie="diffusion.mov")  # create movie tracker

eq = DiffusionPDE()  # define the physics
eq.solve(state, t_range=2, dt=0.005, tracker=tracker)
