#!/usr/bin/env python3
"""
This script tests the performance of different solvers
"""

import sys
from pathlib import Path
from typing import Literal

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))

import numpy as np

from pde import CahnHilliardPDE, Controller, DiffusionPDE, ScalarField, UnitGrid
from pde.solvers import CrankNicolsonSolver, ExplicitSolver, ImplicitSolver, ScipySolver


def main(
    equation: Literal["diffusion", "cahn-hilliard"] = "cahn-hilliard",
    t_range: float = 100,
    size: int = 32,
):
    """main routine testing the performance

    Args:
        equation (str):
            Chooses the equation to consider
        t_range (float):
            Sets the total duration that should be solved for
        size (int):
            The number of grid points along each axis
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
    print(f"EQUATION: ∂c/∂t = {eq.expression}\n")

    print("Determine ground truth...")
    expected = eq.solve(field, t_range=t_range, dt=1e-4, tracker=["progress"])

    print("\nSOLVER PERFORMANCE:")
    solvers = {
        "Euler, fixed": (1e-3, ExplicitSolver(eq, scheme="euler", adaptive=False)),
        "Euler, adaptive": (1e-3, ExplicitSolver(eq, scheme="euler", adaptive=True)),
        "Runge-Kutta, fixed": (1e-2, ExplicitSolver(eq, scheme="rk", adaptive=False)),
        "Runge-Kutta, adaptive": (1e-2, ExplicitSolver(eq, scheme="rk", adaptive=True)),
        "Implicit": (1e-2, ImplicitSolver(eq)),
        "Crank-Nicolson": (1e-2, CrankNicolsonSolver(eq)),
        "Scipy": (None, ScipySolver(eq)),
    }

    for name, (dt, solver) in solvers.items():
        # run the simulation with the given solver
        solver.backend = "numba"
        controller = Controller(solver, t_range=t_range, tracker=None)
        result = controller.run(field, dt=dt)

        # determine the deviation from the ground truth
        error = np.linalg.norm(result.data - expected.data)
        error_str = f"{error:.4g}"

        # report the runtime
        runtime_str = f"{controller.info['profiler']['solver']:.3g}"

        # report information about the time step
        if solver.info.get("dt_adaptive", False):
            stats = solver.info["dt_statistics"]
            dt_str = f"{stats['min']:.3g} .. {stats['max']:.3g}"
        elif solver.info["dt"] is None:
            dt_str = "automatic"
        else:
            dt_str = f"{solver.info['dt']:.3g}"

        print(f"{name:>21s}:  runtime={runtime_str:8} error={error_str:11} dt={dt_str}")


if __name__ == "__main__":
    main()
