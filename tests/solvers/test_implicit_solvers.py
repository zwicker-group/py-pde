"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDE, DiffusionPDE, ScalarField, UnitGrid
from pde.solvers import Controller, ImplicitSolver
from pde.tools import mpi


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_implicit_solvers_simple_fixed(backend):
    """test implicit solvers"""
    grid = UnitGrid([4])
    xs = grid.axes_coords[0]
    field = ScalarField.from_expression(grid, "x")
    eq = PDE({"c": "c"})

    dt = 0.01
    solver = ImplicitSolver(eq, backend=backend)
    controller = Controller(solver, t_range=10.0, tracker=None)
    res = controller.run(field, dt=dt)

    if mpi.is_main:
        np.testing.assert_allclose(res.data, xs * np.exp(10), rtol=0.1)
        assert solver.info["steps"] == pytest.approx(10 / dt, abs=1)
        assert not solver.info.get("dt_adaptive", False)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_implicit_stochastic_solvers(backend, rng):
    """test simple version of the stochastic implicit solver"""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE()
    seq = DiffusionPDE(noise=1e-10)

    solver1 = ImplicitSolver(eq, backend=backend)
    c1 = Controller(solver1, t_range=1, tracker=None)
    s1 = c1.run(field, dt=1e-3)

    solver2 = ImplicitSolver(seq, backend=backend)
    c2 = Controller(solver2, t_range=1, tracker=None)
    s2 = c2.run(field, dt=1e-3)

    np.testing.assert_allclose(s1.data, s2.data, rtol=1e-4, atol=1e-4)
    assert not solver1.info["stochastic"]
    assert solver2.info["stochastic"]

    assert not solver1.info.get("dt_adaptive", False)
    assert not solver2.info.get("dt_adaptive", False)
