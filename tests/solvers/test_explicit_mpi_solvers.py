"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDE, DiffusionPDE, FieldCollection, ScalarField, UnitGrid
from pde.solvers import Controller, ExplicitMPISolver
from pde.tools import mpi


@pytest.mark.multiprocessing
@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize(
    "scheme, adaptive, decomposition",
    [
        ("euler", False, "auto"),
        ("euler", True, [1, -1]),
        ("runge-kutta", True, [-1, 1]),
    ],
)
def test_simple_pde_mpi(backend, scheme, adaptive, decomposition, rng):
    """Test setting boundary conditions using numba."""
    grid = UnitGrid([8, 8], periodic=[True, False])

    field = ScalarField.random_uniform(grid, rng=rng)
    eq = DiffusionPDE()

    args = {
        "state": field,
        "t_range": 1.01,
        "dt": 0.1,
        "adaptive": adaptive,
        "scheme": scheme,
        "tracker": None,
        "ret_info": True,
    }
    res_mpi, info_mpi = eq.solve(
        backend=backend, solver="explicit_mpi", decomposition=decomposition, **args
    )

    if mpi.is_main:
        # check results in the main process
        expect, info2 = eq.solve(backend="numpy", solver="explicit", **args)
        np.testing.assert_allclose(res_mpi.data, expect.data)

        assert info_mpi["solver"]["steps"] == info2["solver"]["steps"]
        assert info_mpi["solver"]["use_mpi"]
        if decomposition != "auto":
            for i in range(2):
                if decomposition[i] == 1:
                    assert info_mpi["solver"]["grid_decomposition"][i] == 1
                else:
                    assert info_mpi["solver"]["grid_decomposition"][i] == mpi.size


@pytest.mark.multiprocessing
@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_stochastic_mpi_solvers(backend, rng):
    """Test simple version of the stochastic solver."""
    field = ScalarField.random_uniform(UnitGrid([16]), -1, 1, rng=rng)
    eq = DiffusionPDE()
    seq = DiffusionPDE(noise=1e-10)

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


@pytest.mark.multiprocessing
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_multiple_pdes_mpi(backend, rng):
    """Test setting boundary conditions using numba."""
    grid = UnitGrid([8, 8], periodic=[True, False])

    fields = FieldCollection.scalar_random_uniform(2, grid, rng=rng)
    eq = PDE({"a": "laplace(a) - b", "b": "laplace(b) + a"})

    args = {
        "state": fields,
        "t_range": 1.01,
        "dt": 0.1,
        "adaptive": True,
        "scheme": "euler",
        "tracker": None,
        "ret_info": True,
    }
    res_mpi, info_mpi = eq.solve(backend=backend, solver="explicit_mpi", **args)

    if mpi.is_main:
        # check results in the main process
        expect, info2 = eq.solve(backend="numpy", solver="explicit", **args)
        np.testing.assert_allclose(res_mpi.data, expect.data)

        assert info_mpi["solver"]["steps"] == info2["solver"]["steps"]
        assert info_mpi["solver"]["use_mpi"]
