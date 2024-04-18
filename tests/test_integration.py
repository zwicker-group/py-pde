"""
Integration tests that use multiple modules together

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from pde import CartesianGrid, DiffusionPDE, FileStorage, PDEBase, ScalarField, UnitGrid
from pde.tools import misc, mpi, numba
from pde.tools.misc import module_available


@pytest.mark.skipif(not misc.module_available("h5py"), reason="requires `h5py` module")
def test_writing_to_storage(tmp_path, rng):
    """test whether data is written to storage"""
    state = ScalarField.random_uniform(UnitGrid([3]), rng=rng)
    pde = DiffusionPDE()
    path = tmp_path / "test_writing_to_storage.hdf5"
    data = FileStorage(filename=path)
    pde.solve(state, t_range=1.1, dt=0.1, tracker=[data.tracker(0.5)])

    assert len(data) == 3


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_inhomogeneous_bcs_1(backend):
    """test simulation with inhomogeneous boundary conditions"""
    grid = CartesianGrid([[0, 2 * np.pi], [0, 1]], [32, 2], periodic=[True, False])
    state = ScalarField(grid)
    eq = DiffusionPDE(bc=["auto_periodic_neumann", {"value": "sin(x)"}])
    sol = eq.solve(state, t_range=1e1, dt=1e-2, backend=backend, tracker=None)
    data = sol.get_line_data(extract="project_x")
    np.testing.assert_almost_equal(
        data["data_y"], 0.9 * np.sin(data["data_x"]), decimal=2
    )


def test_inhomogeneous_bcs_2():
    """test simulation with inhomogeneous boundary conditions"""
    grid = CartesianGrid([[0, 1], [0, 1]], [8, 8], periodic=False)
    state = ScalarField(grid)
    eq = DiffusionPDE(bc={"value": "x + y"})
    sol = eq.solve(state, t_range=1e1, dt=1e-3, tracker=None)
    expect = ScalarField.from_expression(grid, "x + y")
    np.testing.assert_almost_equal(sol.data, expect.data)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_inhomogeneous_bcs_func(backend):
    """test simulation with inhomogeneous boundary conditions"""
    grid = CartesianGrid([[-5, 5], [-5, 5]], 32)
    field = ScalarField(grid)

    def bc_value(adjacent_value, dx, x, y, t):
        return np.sign(x)

    bc_x = "derivative"
    bc_y = ["derivative", {"value_expression": bc_value}]

    eq = DiffusionPDE(bc=[bc_x, bc_y])
    res = eq.solve(field, t_range=10, dt=0.01, adaptive=True, backend=backend)
    assert np.all(res.data[:16] < 0)
    assert np.all(res.data[16:] > 0)


@pytest.mark.multiprocessing
def test_custom_pde_mpi(caplog, rng):
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
    field = ScalarField.random_uniform(grid, rng=rng)
    eq = TestPDE()

    args = {
        "state": field,
        "t_range": 1.01,
        "dt": 0.1,
        "tracker": None,
        "ret_info": True,
    }

    res1, info1 = eq.solve(backend="numpy", solver="explicit_mpi", **args)
    if mpi.is_main:
        assert "significant state modifications" in caplog.text
        caplog.clear()

    res2, info2 = eq.solve(backend="numba", solver="explicit_mpi", **args)
    if mpi.is_main:
        assert "significant state modifications" in caplog.text

    if mpi.is_main:
        # check results in the main process
        expect, info3 = eq.solve(backend="numpy", solver="explicit", **args)

        np.testing.assert_allclose(res1.data, expect.data)
        np.testing.assert_allclose(res2.data, expect.data)
        assert info3["solver"]["state_modifications"] > 0

        for info in [info1["solver"], info2["solver"]]:
            assert info["steps"] == 11
            assert info["use_mpi"]
            assert info["state_modifications"] == info3["solver"]["state_modifications"]


@pytest.mark.skipif(sys.platform == "win32", reason="submit_job has issues on windows")
@pytest.mark.skipif(
    not module_available("modelrunner"), reason="requires `py-modelrunner`"
)
def test_modelrunner_storage_one(tmp_path, capsys):
    """test how modelrunner storage can be used"""
    import modelrunner as mr

    SCRIPT_PATH = Path(__file__).parent / "resources"
    assert SCRIPT_PATH.is_dir()
    output = tmp_path / "result.yaml"  # TODO: Change back to JSON
    assert not output.is_file()

    outs, errs = mr.submit_job(
        SCRIPT_PATH / "run_pde.py",
        output=output,
        # log_folder=tmp_path,
        parameters={"t_range": 1.5},
        method="foreground",
        overwrite_strategy="error",
    )
    assert outs == errs == ""
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    assert output.is_file()

    print("=" * 40)
    print(open(output).read())
    print("=" * 40)

    # read storage manually
    with mr.open_storage(output, mode="read") as storage_obj:
        np.testing.assert_allclose(storage_obj["storage/trajectory"].times, [0, 1])
        assert isinstance(storage_obj["storage/initial_state"], ScalarField)

    # read storage using `Result` class
    result = mr.Result.from_file(output)
    assert isinstance(result.result["field"], ScalarField)
    assert isinstance(result.storage["initial_state"], ScalarField)
    np.testing.assert_allclose(result.storage["trajectory"].times, [0, 1])

    # delete temporary files
    for path in tmp_path.iterdir():
        path.unlink()


@pytest.mark.skipif(sys.platform == "win32", reason="submit_jobs has issues on windows")
@pytest.mark.skipif(
    not module_available("modelrunner"), reason="requires `py-modelrunner`"
)
def test_modelrunner_storage_many(tmp_path):
    """test how modelrunner storage can be used"""
    import modelrunner as mr

    SCRIPT_PATH = Path(__file__).parent / "resources"
    assert SCRIPT_PATH.is_dir()

    num_jobs = mr.submit_jobs(
        SCRIPT_PATH / "run_pde.py",
        output_folder=tmp_path,
        parameters={"t_range": [1.5, 2.5]},
        log_folder=tmp_path,
        method="foreground",
        overwrite_strategy="error",
    )
    assert num_jobs == 2

    # read storage manually
    for path in tmp_path.iterdir():
        if path.is_file() and not path.name.endswith("txt"):
            with mr.open_storage(path) as storage:
                assert "initial_state" in storage["storage"].keys()
                assert "trajectory" in storage["storage"].keys()
                assert "result" in storage.keys()

    # read result using ResultCollection
    results = mr.ResultCollection.from_folder(tmp_path)
    assert len(results) == num_jobs
    for result in results:
        t_range = result.model.parameters["t_range"]
        assert isinstance(result.result["field"], ScalarField)
        assert isinstance(result.storage["initial_state"], ScalarField)
        np.testing.assert_allclose(
            result.storage["trajectory"].times, range(int(t_range) + 1)
        )

    # delete temporary files
    for path in tmp_path.iterdir():
        path.unlink()
