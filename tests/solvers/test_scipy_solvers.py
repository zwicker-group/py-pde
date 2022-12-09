"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde import DiffusionPDE, ScalarField, UnitGrid
from pde.solvers import Controller, ScipySolver


def test_no_dt():
    """test scipy solver without timestep"""
    grid = UnitGrid([16])
    field = ScalarField.random_uniform(grid, -1, 1)
    eq = DiffusionPDE()

    c1 = Controller(ScipySolver(eq), t_range=1, tracker=None)
    s1 = c1.run(field, dt=1e-3)

    c2 = Controller(ScipySolver(eq), t_range=1, tracker=None)
    s2 = c2.run(field)

    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-3, atol=1e-3)
