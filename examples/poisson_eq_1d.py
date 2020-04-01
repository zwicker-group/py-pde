"""
Solving Poisson's equation in 1d
================================

This example shows how to solve a 1d Poisson equation with boundary conditions.
"""

from pde import CartesianGrid, ScalarField

grid = CartesianGrid([[0, 1]], 16, periodic=False)
field = ScalarField(grid, 1)
result = field.solve_poisson([{'value': 0}, {'derivative': 1}])

result.plot()
