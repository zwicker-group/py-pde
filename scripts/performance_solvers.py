#!/usr/bin/env python3
"""This script tests the performance of different solvers."""

import sys
from pathlib import Path
from typing import Literal

PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))

import numpy as np

from pde import CahnHilliardPDE, Controller, DiffusionPDE, ScalarField, UnitGrid
from pde.solvers import (
    AdamsBashforthSolver,
    CrankNicolsonSolver,
    EulerSolver,
    ImplicitSolver,
    RungeKuttaSolver,
    ScipySolver,
)


def main(
    equation: Literal["diffusion", "cahn-hilliard"] = "cahn-hilliard",
    t_range: float = 100,
    size: int = 32,
    *,
    backends: tuple[str] = ("auto",),
):
    """Main routine testing the performance.

    Args:
        equation (str):
            Chooses the equation to consider
        t_range (float):
            Sets the total duration that should be solved for
        size (int):
            The number of grid points along each axis
        backend (str):
            The backend to use
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
        msg = f"Undefined equation `{equation}`"
        raise ValueError(msg)
    print(f"EQUATION: ∂c/∂t = {eq.expression}\n")

    print("Determine ground truth...")
    expected = eq.solve(field, t_range=t_range, dt=1e-4, tracker=["progress"])

    for backend in backends:
        print(f"\nBACKEND={backend}:")

        # define all solvers to be tested
        solvers = {
            "Euler, fixed": (1e-3, EulerSolver(eq, adaptive=False, backend=backend)),
            "Euler, adaptive": (1e-3, EulerSolver(eq, adaptive=True, backend=backend)),
            "Runge-Kutta, fixed": (
                1e-2,
                RungeKuttaSolver(eq, adaptive=False, backend=backend),
            ),
            "Runge-Kutta, adaptive": (
                1e-2,
                RungeKuttaSolver(eq, adaptive=True, backend=backend),
            ),
            "Implicit": (1e-2, ImplicitSolver(eq, backend=backend)),
            "Adams-Bashforth": (1e-2, AdamsBashforthSolver(eq, backend=backend)),
            "Crank-Nicolson": (1e-2, CrankNicolsonSolver(eq, backend=backend)),
            "Scipy": (None, ScipySolver(eq, backend=backend)),
        }

        for name, (dt, solver) in solvers.items():
            # run the simulation with the given solver
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

            print(
                f"{name:>23s}: runtime={runtime_str:8} error={error_str:11} dt={dt_str}"
            )


if __name__ == "__main__":
    main(t_range=10, size=512, backends=("numba", "torch"))
