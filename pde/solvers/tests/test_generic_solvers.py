"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde import PDE, DiffusionPDE, FieldCollection, ScalarField, UnitGrid
from pde.solvers import (
    Controller,
    ExplicitSolver,
    ImplicitSolver,
    ScipySolver,
    registered_solvers,
)


def test_solver_registration():
    """ test solver registration """
    solvers = registered_solvers()
    assert "explicit" in solvers
    assert "scipy" in solvers


def test_solver_in_pde_class():
    """ test whether solver instances can be used in pde instances """
    field = ScalarField.random_uniform(UnitGrid([16, 16]), -1, 1)
    eq = DiffusionPDE()
    eq.solve(field, t_range=1, method=ScipySolver, tracker=None)


@pytest.mark.parametrize("solver_class", [ExplicitSolver, ImplicitSolver, ScipySolver])
def test_compare_solvers(solver_class):
    """ compare several solvers """
    field = ScalarField.random_uniform(UnitGrid([8, 8]), -1, 1)
    eq = DiffusionPDE()

    # ground truth
    c1 = Controller(ExplicitSolver(eq, scheme="runge-kutta"), t_range=0.1, tracker=None)
    s1 = c1.run(field, dt=5e-3)

    c2 = Controller(solver_class(eq), t_range=0.1, tracker=None)
    with np.errstate(under="ignore"):
        s2 = c2.run(field, dt=5e-3)

    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_solvers_complex(backend):
    """ test solvers with a complex PDE """
    r = FieldCollection.scalar_random_uniform(2, UnitGrid([3]), labels=["a", "b"])
    c = r["a"] + 1j * r["b"]
    assert c.is_complex

    # assume c = a + i * b
    eq_c = PDE({"c": "-I * laplace(c)"})
    eq_r = PDE({"a": "laplace(b)", "b": "-laplace(a)"})
    res_r = eq_r.solve(r, t_range=1e-2, dt=1e-3, backend="numpy", tracker=None)
    exp_c = res_r[0].data + 1j * res_r[1].data

    for solver_class in [ExplicitSolver, ImplicitSolver, ScipySolver]:
        solver = solver_class(eq_c, backend=backend)
        controller = Controller(solver, t_range=1e-2, tracker=None)
        res_c = controller.run(c, dt=1e-3)
        np.testing.assert_allclose(res_c.data, exp_c, rtol=1e-3, atol=1e-3)
