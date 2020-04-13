"""
Solver comparison
=================

This example shows how to set up solvers explicitly and how to extract
diagnostic information.
"""

from pde import (UnitGrid, ScalarField, FieldCollection, DiffusionPDE,
                 ExplicitSolver, ScipySolver, Controller) 

# initialize the grid, an initial condition, and the PDE
grid = UnitGrid([32, 32])
field = ScalarField.random_uniform(grid, -1, 1)
eq = DiffusionPDE()

# try the explicit solver
solver1 = ExplicitSolver(eq)
controller1 = Controller(solver1, t_range=1, tracker=None)
sol1 = controller1.run(field, dt=1e-3)
sol1.label = 'explicit solver'
print('Diagnostic information from first run:')
print(controller1.diagnostics)
print()

# try the standard scipy solver
solver2 = ScipySolver(eq)
controller2 = Controller(solver2, t_range=1, tracker=None)
sol2 = controller2.run(field)
sol2.label = 'scipy solver'
print('Diagnostic information from second run:')
print(controller2.diagnostics)
print()

# plot both fields and give the deviation as the title
title = f'Deviation: {((sol1 - sol2)**2).average:.2g}'
FieldCollection([sol1, sol2]).plot(title=title, show=True)
