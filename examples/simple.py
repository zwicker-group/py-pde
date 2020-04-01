"""
Simple diffusion equation
=========================

This example solves a simple diffusion equation in two dimensions.
"""

from pde import DiffusionPDE, UnitGrid, ScalarField

grid = UnitGrid([64, 64])  # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

eq = DiffusionPDE(diffusivity=0.1)  # define the pde
result = eq.solve(state, t_range=10)
result.plot()
