"""
Integration tests that use multiple modules together

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, DiffusionPDE, FileStorage, PDEBase, ScalarField, UnitGrid
from pde.tools import misc, mpi, numba


@misc.skipUnlessModule("h5py")
def test_writing_to_storage(tmp_path):
    """test whether data is written to storage"""
    state = ScalarField.random_uniform(UnitGrid([3]))
    pde = DiffusionPDE()
    path = tmp_path / "test_writing_to_storage.hdf5"
    data = FileStorage(filename=path)
    pde.solve(state, t_range=1.1, dt=0.1, tracker=[data.tracker(0.5)])

    assert len(data) == 3


def test_inhomogeneous_bcs():
    """test simulation with inhomogeneous boundary conditions"""
    # single coordinate
    grid = CartesianGrid([[0, 2 * np.pi], [0, 1]], [32, 2], periodic=[True, False])
    state = ScalarField(grid)
    pde = DiffusionPDE(
        bc=["auto_periodic_neumann", {"type": "value", "value": "sin(x)"}]
    )
    sol = pde.solve(state, t_range=1e1, dt=1e-2, tracker=None)
    data = sol.get_line_data(extract="project_x")
    np.testing.assert_almost_equal(
        data["data_y"], 0.9 * np.sin(data["data_x"]), decimal=2
    )

    # double coordinate
    grid = CartesianGrid([[0, 1], [0, 1]], [8, 8], periodic=False)
    state = ScalarField(grid)
    pde = DiffusionPDE(bc={"type": "value", "value": "x + y"})
    sol = pde.solve(state, t_range=1e1, dt=1e-3, tracker=None)
    expect = ScalarField.from_expression(grid, "x + y")
    np.testing.assert_almost_equal(sol.data, expect.data)


@pytest.mark.multiprocessing
def test_custom_pde_mpi(caplog):
    """test a custom PDE using the parallelized solver"""

    class TestPDE(PDEBase):
        def make_modify_after_step(self, state):
            def modify_after_step(state_data):
                modification = 0
                for i in range(state_data.size):
                    if state_data.flat[i] > 1:
                        state_data.flat[i] -= 1
                        modification += 2
                return modification

            return modify_after_step

        def evolution_rate(self, state, t=0):
            return ScalarField(state.grid, 1)

        def _make_pde_rhs_numba(self, state):
            @numba.jit
            def pde_rhs(state_data, t):
                return np.ones_like(state_data)

            return pde_rhs

    grid = UnitGrid([16, 16])
    field = ScalarField.random_uniform(grid)
    eq = TestPDE()

    args = {
        "state": field,
        "t_range": 1.01,
        "dt": 0.1,
        "tracker": None,
        "ret_info": True,
    }

    res1, info1 = eq.solve(backend="numpy", method="explicit_mpi", **args)
    if mpi.is_main:
        assert "significant state modifications" in caplog.text
        caplog.clear()

    res2, info2 = eq.solve(backend="numba", method="explicit_mpi", **args)
    if mpi.is_main:
        assert "significant state modifications" in caplog.text

    if mpi.is_main:
        # check results in the main process
        expect, info3 = eq.solve(backend="numpy", method="explicit", **args)

        np.testing.assert_allclose(res1.data, expect.data)
        np.testing.assert_allclose(res2.data, expect.data)
        assert info3["solver"]["state_modifications"] > 0

        for info in [info1["solver"], info2["solver"]]:
            assert info["steps"] == 11
            assert info["use_mpi"]
            assert info["state_modifications"] == info3["solver"]["state_modifications"]
