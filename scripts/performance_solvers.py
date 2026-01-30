#!/usr/bin/env python3
"""This script tests the performance of different solvers."""

import sys
from pathlib import Path
from typing import Literal

# determine path of the `py-pde` package
PACKAGE_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_PATH))

# determine path of the cache for the ground truth of simulations
GROUND_TRUTH_CACHE = Path(__file__).resolve().parent / "_cache" / "solver_ground_truth"

import numpy as np

from pde import (
    CahnHilliardPDE,
    Controller,
    DiffusionPDE,
    PDEBase,
    ScalarField,
    UnitGrid,
)
from pde.solvers import (
    AdamsBashforthSolver,
    EulerSolver,
    RungeKuttaSolver,
    ScipySolver,
)


def get_problem(
    equation: Literal["diffusion", "cahn-hilliard"] = "cahn-hilliard",
    t_range: float = 100,
    size: int = 32,
) -> tuple[PDEBase, ScalarField]:
    """Define the problem we're trying to solve.

    Args:
        equation (str):
            Chooses the equation to consider
        t_range (float):
            Sets the total duration that should be solved for
        size (int):
            The number of grid points along each axis

    Returns:
        A scalar field
    """
    # initialize the grid and random field
    rng = np.random.default_rng(0)
    grid = UnitGrid([size, size], periodic=False)
    field = ScalarField.random_uniform(grid, rng=rng)

    # determine the equation to solve
    if equation == "diffusion":
        eq = DiffusionPDE()
    elif equation == "cahn-hilliard":
        eq = CahnHilliardPDE()
    else:
        msg = f"Undefined equation `{equation}`"
        raise ValueError(msg)
    return eq, field


def get_ground_truth(
    equation: Literal["diffusion", "cahn-hilliard"] = "cahn-hilliard",
    t_range: float = 100,
    size: int = 32,
) -> ScalarField:
    """Generate the ground truth for the setup.

    Args:
        equation (str):
            Chooses the equation to consider
        t_range (float):
            Sets the total duration that should be solved for
        size (int):
            The number of grid points along each axis
    """
    cache_file = GROUND_TRUTH_CACHE / f"{equation}_{t_range}_{size}.hdf"

    if cache_file.exists():
        # read ground truth to cache
        print(f"Loading ground truth from cache: {cache_file}")
        expected = ScalarField.from_file(cache_file)

    else:
        # calculate ground truth
        print("Determine ground truth...")
        eq, field = get_problem(equation, t_range, size)
        expected = eq.solve(field, t_range=t_range, dt=1e-4, tracker=["progress"])

        # write ground truth to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        expected.to_file(cache_file)

    return expected


def main(
    equation: Literal["diffusion", "cahn-hilliard"] = "cahn-hilliard",
    t_range: float = 100,
    size: int = 32,
    *,
    backends: tuple[Literal["auto", "numba", "torch"]] = ("auto",),
) -> None:
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

    # determine grid, initial state, and the PDE
    eq, field = get_problem(equation, t_range, size)
    print(f"GRID: {field.grid}")
    print(f"EQUATION: ∂c/∂t = {eq.expression}")
    print(f"TIME RANGE: t_range={t_range}\n")

    # determine the ground truth
    expected = get_ground_truth(equation, t_range, size)

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
            # "Implicit": (1e-2, ImplicitSolver(eq, backend=backend)),
            "Adams-Bashforth": (1e-2, AdamsBashforthSolver(eq, backend=backend)),
            # "Crank-Nicolson": (1e-2, CrankNicolsonSolver(eq, backend=backend)),
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
    main(equation="cahn-hilliard", t_range=10, size=512, backends=["torch"])


# REFERENCE MEASUREMENTS
#
# The following measurements were done on a Macbook with Apple M4 Pro CPU
#
# GRID: UnitGrid(shape=(512, 512), periodic=[False, False])
# EQUATION: ∂c/∂t = ∇²(c³ - c - ∇²c)
# TIME RANGE: t_range=10
#
# BACKEND=numba:
#            Euler, fixed: runtime=43.7     error=0.001129    dt=0.001
#         Euler, adaptive: runtime=3.91     error=0.02308     dt=0.00044 .. 0.116
#      Runge-Kutta, fixed: runtime=19.8     error=0.0001643   dt=0.01
#   Runge-Kutta, adaptive: runtime=8.25     error=0.000292    dt=0.01 .. 0.0733
#                Implicit: runtime=11.6     error=0.01193     dt=0.01
#         Adams-Bashforth: runtime=13.3     error=0.008489    dt=0.01
#          Crank-Nicolson: runtime=16.9     error=0.001136    dt=0.01
#                   Scipy: runtime=9.35     error=0.05862     dt=automatic
#
# BACKEND=torch:
#            Euler, fixed: runtime=38       error=0.00119     dt=0.001
#         Euler, adaptive: runtime=3.63     error=0.02405     dt=0.000412 .. 0.138
#      Runge-Kutta, fixed: runtime=17.1     error=0.0001651   dt=0.01
#   Runge-Kutta, adaptive: runtime=6.52     error=0.000277    dt=0.01 .. 0.0731
#                Implicit: runtime=122      error=0.01245     dt=0.01
#         Adams-Bashforth: runtime=8.17     error=0.008977    dt=0.01
#          Crank-Nicolson: runtime=120      error=0.001201    dt=0.01
#                   Scipy: runtime=7.06     error=0.1588      dt=automatic
