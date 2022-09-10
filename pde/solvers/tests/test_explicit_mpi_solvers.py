"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import DiffusionPDE, ScalarField, UnitGrid
from pde.solvers import Controller, ExplicitMPISolver
from pde.tools import mpi


@pytest.mark.multiprocessing
@pytest.mark.parametrize(
    "scheme, decomposition", [("euler", [1, -1]), ("runge-kutta", [-1, 1])]
)
def test_simple_pde_mpi(scheme, decomposition):
    """test setting boundary conditions using numba"""
    grid = UnitGrid([8, 8], periodic=[True, False])

    field = ScalarField.random_uniform(grid)
    eq = DiffusionPDE()

    args = {
        "state": field,
        "t_range": 1.01,
        "dt": 0.1,
        "scheme": scheme,
        "tracker": None,
        "ret_info": True,
    }
    res1, info1 = eq.solve(
        backend="numpy", method="explicit_mpi", decomposition=decomposition, **args
    )
    res2, info2 = eq.solve(
        backend="numba", method="explicit_mpi", decomposition=decomposition, **args
    )

    if mpi.is_main:
        # check results in the main process
        expect, _ = eq.solve(backend="numpy", method="explicit", **args)
        np.testing.assert_allclose(res1.data, expect.data)
        np.testing.assert_allclose(res2.data, expect.data)

        for info in [info1, info2]:
            assert info["solver"]["steps"] == 11
            assert info["solver"]["use_mpi"]
            for i in range(2):
                if decomposition[i] == 1:
                    assert info["solver"]["grid_decomposition"][i] == 1
                else:
                    assert info["solver"]["grid_decomposition"][i] == mpi.size


@pytest.mark.multiprocessing
@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_stochastic_mpi_solvers(backend):
    """test simple version of the stochastic solver"""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1)
    eq = DiffusionPDE()
    seq = DiffusionPDE(noise=1e-6)

    solver1 = ExplicitMPISolver(eq, backend=backend)
    c1 = Controller(solver1, t_range=1, tracker=None)
    s1 = c1.run(field, dt=1e-3)

    solver2 = ExplicitMPISolver(seq, backend=backend)
    c2 = Controller(solver2, t_range=1, tracker=None)
    s2 = c2.run(field, dt=1e-3)

    if mpi.is_main:
        np.testing.assert_allclose(s1.data, s2.data, rtol=1e-4, atol=1e-4)
        assert not solver1.info["stochastic"]
        assert solver2.info["stochastic"]

        assert not solver1.info["dt_adaptive"]
        assert not solver2.info["dt_adaptive"]
