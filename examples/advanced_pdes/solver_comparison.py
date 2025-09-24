"""
Solver comparison
=================

This example shows how to set up solvers explicitly and how to extract diagnostic
information.
"""

import pde

# initialize the grid, an initial condition, and the PDE
grid = pde.UnitGrid([32, 32])
field = pde.ScalarField.random_uniform(grid, -1, 1)
eq = pde.DiffusionPDE()


def run_solver(solver, label):
    """Helper function testing the solver."""
    controller = pde.Controller(solver, t_range=1, tracker=None)
    sol = controller.run(field, dt=1e-3)
    sol.label = label + " solver"
    print(f"Diagnostic information for {sol.label}:")
    print(controller.diagnostics)
    print()
    return sol


# try different solvers
solutions = [
    run_solver(pde.ExplicitSolver(eq), "explicit Euler"),
    run_solver(
        pde.ExplicitSolver(eq, scheme="runge-kutta", adaptive=True),
        "explicit, adaptive Runge-Kutta",
    ),
    run_solver(pde.ImplicitSolver(eq), "implicit"),
    run_solver(pde.CrankNicolsonSolver(eq), "Crank-Nicolson"),
    run_solver(pde.AdamsBashforthSolver(eq), "Adam-Bashforth"),
    run_solver(pde.ScipySolver(eq), "scipy"),
]

# plot both fields and give the deviation as the title
deviations = [(solutions[0] - sol).fluctuations for sol in solutions]
title = "Deviation: " + ", ".join(f"{deviation:.2g}" for deviation in deviations[1:])
pde.FieldCollection(solutions).plot(title=title, arrangement=(2, 3))
