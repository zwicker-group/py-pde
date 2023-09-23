"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde import PDE, DiffusionPDE, FieldCollection, ScalarField, UnitGrid
from pde.solvers import Controller, ScipySolver


def test_scipy_no_dt(rng):
    """test scipy solver without timestep"""
    grid = UnitGrid([16])
    field = ScalarField.random_uniform(grid, -1, 1, rng=rng)
    eq = DiffusionPDE()

    c1 = Controller(ScipySolver(eq), t_range=1, tracker=None)
    s1 = c1.run(field, dt=1e-3)

    c2 = Controller(ScipySolver(eq), t_range=1, tracker=None)
    s2 = c2.run(field)

    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-3, atol=1e-3)


def test_scipy_field_collection():
    """test scipy solver with field collection"""
    grid = UnitGrid([2])
    field = FieldCollection.from_scalar_expressions(grid, ["x", "0"])
    eq = PDE({"a": "1", "b": "a"})

    res = eq.solve(field, t_range=1, dt=1e-2, solver="scipy")
    np.testing.assert_allclose(res.data, np.array([[1.5, 2.5], [1.0, 2.0]]))
