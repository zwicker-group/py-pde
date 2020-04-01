"""
Setting boundary conditions
===========================

This example shows how different boundary conditions can be specified.
"""

from pde import UnitGrid, ScalarField, DiffusionPDE

grid = UnitGrid([16, 16], periodic=[False, True]) # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

# set boundary conditions `bc` for all axes
bc_x_left = {'derivative': 0.1}  
bc_x_right = {'value': 'sin(y)'}
bc_x = [bc_x_left, bc_x_right]
bc_y = 'periodic' 
eq = DiffusionPDE(bc=[bc_x, bc_y])

result = eq.solve(state, t_range=10, dt=0.005)
result.plot(show=True)
