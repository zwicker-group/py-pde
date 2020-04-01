"""
Diffusion on a Cartesian grid
=============================

This example shows how to solve the diffusion equation on a Cartesian grid.
"""

from pde import CartesianGrid, ScalarField, DiffusionPDE

grid = CartesianGrid([[-1, 1], [0, 2]], [15, 8])  # generate grid
state = ScalarField(grid)  # generate initial condition
state.add_interpolated([0, 1], 1)

eq = DiffusionPDE()  # define the pde
result = eq.solve(state, t_range=1, dt=0.005)
result.plot(show=True)
