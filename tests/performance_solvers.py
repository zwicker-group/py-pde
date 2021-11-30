#!/usr/bin/env python3
"""
This script tests the performance of different solvers
"""

import sys
from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))

import numpy as np

from pde import CahnHilliardPDE, Controller, DiffusionPDE, ScalarField, UnitGrid
from pde.solvers import ExplicitSolver, ImplicitSolver, ScipySolver


def main(equation: str = "cahn-hilliard", t_range: float = 100, size: int = 32):
    """main routine testing the performance

    Args:
        equation (str): Chooses the equation to consider
        t_range (float): Sets the total duration that should be solved for
        size (int): The number of grid points along each axis
    """
    print("Reports duration in seconds (smaller is better)\n")

    # determine grid and initial state
    grid = UnitGrid([size, size], periodic=False)
    field = ScalarField.random_uniform(grid)
    print(f"GRID: {grid}")

    # determine the equation to solve
    if equation == "diffusion":
        eq = DiffusionPDE()
    elif equation == "cahn-hilliard":
        eq = CahnHilliardPDE()
    else:
        raise ValueError(f"Undefined equation `{equation}`")
    print(f"EQUATION: ∂c/∂t = {eq.expression}")

    print("\nSOLVER PERFORMANCE:")

    expected = eq.solve(field, t_range=t_range, dt=1e-5, tracker=None)

    solvers = {
        "Euler, fixed": (1e-3, ExplicitSolver(eq, scheme="euler", adaptive=False)),
        "Euler, adaptive": (1e-3, ExplicitSolver(eq, scheme="euler", adaptive=True)),
        "Runge-Kutta, fixed": (1e-3, ExplicitSolver(eq, scheme="rk", adaptive=False)),
        "Runge-Kutta, adaptive": (1e-3, ExplicitSolver(eq, scheme="rk", adaptive=True)),
        "implicit": (1e-3, ImplicitSolver(eq)),
        "scipy": (None, ScipySolver(eq)),
    }

    for name, (dt, solver) in solvers.items():

        solver.backend = "numba"
        controller = Controller(solver, t_range=t_range, tracker=None)
        result = controller.run(field, dt=dt)

        # call once to pre-compile and test result
        if np.allclose(result.data, expected.data, atol=1e-2):
            # report the duration
            print(f"{name:>21s}: {controller.info['profiler']['solver']:.3g}")
        else:
            # report the mismatch
            print(f"{name:>21s}: MISMATCH")


if __name__ == "__main__":
    main()
