"""
Klein-Gordon equation
=====================

This example solves the Klein-Gordon equation in one dimension, showing the
effect of the mass term on wave propagation.  With mass=0 the equation reduces
to the standard wave equation; increasing the mass introduces dispersion.
"""

from pde import KleinGordonPDE, ScalarField, UnitGrid

grid = UnitGrid([128])  # generate grid
u = ScalarField.from_expression(grid, "exp(-((x - 32) / 5) ** 2)")

eq = KleinGordonPDE(speed=1, mass=0.5)  # define the pde
state = eq.get_initial_condition(u)
result = eq.solve(state, t_range=50, dt=0.01)
result[0].plot()
