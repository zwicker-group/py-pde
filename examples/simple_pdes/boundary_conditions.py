"""
Setting boundary conditions
===========================

This example shows how different boundary conditions can be specified.
"""

from pde import DiffusionPDE, ScalarField, UnitGrid

grid = UnitGrid([32, 32], periodic=[False, True])  # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

# set boundary conditions `bc` for all axes
eq = DiffusionPDE(
    bc={"x-": {"derivative": 0.1}, "x+": {"value": "sin(y / 2)"}, "y": "periodic"}
)

result = eq.solve(state, t_range=10, dt=0.005)
result.plot()
