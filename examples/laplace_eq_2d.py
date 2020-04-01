"""
Solving Laplace's equation in 2d
================================

This example shows how to solve a 2d Laplace equation with spatially varying
boundary conditions.
"""

import numpy as np
from pde import CartesianGrid, ScalarField

grid = CartesianGrid([[0, 2 * np.pi]] * 2, 16)
bcs = [{'value': 'sin(y)'}, {'value': 'sin(x)'}]

field = ScalarField(grid)
res = field.solve_poisson(bcs)
res.plot()
