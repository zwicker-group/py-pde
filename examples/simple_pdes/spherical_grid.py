"""
Spherically symmetric PDE
=========================

This example illustrates how to solve a PDE in a spherically symmetric geometry.
"""

from pde import DiffusionPDE, ScalarField, SphericalSymGrid

grid = SphericalSymGrid(radius=[1, 5], shape=128)  # generate grid
state = ScalarField.random_uniform(grid)  # generate initial condition

eq = DiffusionPDE(0.1)  # define the PDE
result = eq.solve(state, t_range=0.1, dt=0.001)

result.plot(kind="image")
