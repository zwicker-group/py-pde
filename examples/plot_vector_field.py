"""
Plotting a vector field
=======================

This example shows how to initialize and visualize a vector field.
"""

from pde import CartesianGrid, VectorField

grid = CartesianGrid([[-2, 2], [-2, 2]], 32)
field = VectorField.from_expression(grid, ['sin(x)', 'cos(x)'])
field.plot(method='streamplot', title='Stream plot of a vector Field')
