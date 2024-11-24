"""
Diffusion on a Cartesian grid
=============================

This example shows how to solve the diffusion equation on a Cartesian grid.
"""

from pde import CartesianGrid, DiffusionPDE, ScalarField

grid = CartesianGrid([[-1, 1], [0, 2]], [30, 16])  # generate grid
state = ScalarField(grid)  # generate initial condition
state.insert([0, 1], 1)

eq = DiffusionPDE(0.1)  # define the pde
result = eq.solve(state, t_range=1, dt=0.01)
result.plot(cmap="magma")
