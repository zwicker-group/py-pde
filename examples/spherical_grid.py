"""
Spherical grid
==============

This example illustrates how a spherical grid can be used.
"""

from pde import DiffusionPDE, SphericalGrid, ScalarField

grid = SphericalGrid(radius=5, shape=16)            # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

eq = DiffusionPDE()                                 # define the pde
result = eq.solve(state, t_range=1, dt=0.005)
result.plot(show=True)
