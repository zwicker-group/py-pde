"""
Spherically symmetric PDE
=========================

This example illustrates how to solve a PDE in a spherically symmetric geometry.
"""

from pde import DiffusionPDE, SphericalGrid, ScalarField

grid = SphericalGrid(radius=[1, 5], shape=32)  # generate grid
state = ScalarField.random_uniform(grid)  # generate initial condition

eq = DiffusionPDE()  # define the PDE
result = eq.solve(state, t_range=0.1, dt=0.005)

result.plot('image', show=True)
