"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDE, DiffusionPDE, FieldCollection, MemoryStorage, ScalarField, UnitGrid
from pde.solvers import (
    Controller,
    CrankNicolsonSolver,
    ExplicitSolver,
    ImplicitSolver,
    ScipySolver,
    registered_solvers,
)
from pde.solvers.base import AdaptiveSolverBase

SOLVER_CLASSES = [ExplicitSolver, ImplicitSolver, CrankNicolsonSolver, ScipySolver]


def test_solver_registration():
    """test solver registration"""
    solvers = registered_solvers()
    assert "explicit" in solvers
    assert "implicit" in solvers
    assert "crank-nicolson" in solvers
    assert "scipy" in solvers


def test_solver_in_pde_class(rng):
    """test whether solver instances can be used in pde instances"""
    field = ScalarField.random_uniform(UnitGrid([16, 16]), -1, 1, rng=rng)
    eq = DiffusionPDE()
    eq.solve(field, t_range=1, solver=ScipySolver, tracker=None)


@pytest.mark.parametrize("solver_class", SOLVER_CLASSES)
def test_compare_solvers(solver_class, rng):
    """compare several solvers"""
    field = ScalarField.random_uniform(UnitGrid([8, 8]), -1, 1, rng=rng)
    eq = DiffusionPDE()

    # ground truth
    c1 = Controller(ExplicitSolver(eq, scheme="runge-kutta"), t_range=0.1, tracker=None)
    s1 = c1.run(field, dt=5e-3)

    c2 = Controller(solver_class(eq), t_range=0.1, tracker=None)
    with np.errstate(under="ignore"):
        s2 = c2.run(field, dt=5e-3)

    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("solver_class", SOLVER_CLASSES)
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_solvers_complex(solver_class, backend):
    """test solvers with a complex PDE"""
    r = FieldCollection.scalar_random_uniform(2, UnitGrid([3]), labels=["a", "b"])
    c = r["a"] + 1j * r["b"]
    assert c.is_complex

    # assume c = a + i * b
    eq_c = PDE({"c": "-I * laplace(c)"})
    eq_r = PDE({"a": "laplace(b)", "b": "-laplace(a)"})
    res_r = eq_r.solve(r, t_range=1e-2, dt=1e-3, backend="numpy", tracker=None)
    exp_c = res_r[0].data + 1j * res_r[1].data

    solver = solver_class(eq_c, backend=backend)
    controller = Controller(solver, t_range=1e-2, tracker=None)
    res_c = controller.run(c, dt=1e-3)
    np.testing.assert_allclose(res_c.data, exp_c, rtol=1e-3, atol=1e-3)


def test_basic_adaptive_solver():
    """test basic adaptive solvers"""
    grid = UnitGrid([4])
    y0 = np.array([1e-3, 1e-3, 1e3, 1e3])
    field = ScalarField(grid, y0)
    eq = PDE({"c": "c"})

    dt = 0.1

    solver = AdaptiveSolverBase(eq, tolerance=1e-1)
    storage = MemoryStorage()
    controller = Controller(solver, t_range=10.1, tracker=storage.tracker(1.0))
    res = controller.run(field, dt=dt)

    np.testing.assert_allclose(res.data, y0 * np.exp(10.1), rtol=0.02)
    assert solver.info["steps"] != pytest.approx(10.1 / dt, abs=1)
    assert solver.info["dt_adaptive"]
    assert solver.info["dt_statistics"]["min"] < 0.0005
    assert np.allclose(storage.times, np.arange(11))
