"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import DiffusionPDE, FieldCollection, ScalarField, Tensor2Field, VectorField
from pde.grids import CylindricalSymGrid, PolarSymGrid, SphericalSymGrid, UnitGrid
from pde.grids._mesh import GridMesh
from pde.tools import mpi

GRIDS = [
    (UnitGrid([3, 4]), [1, -1]),
    (PolarSymGrid(3, 4), [-1]),
    (SphericalSymGrid(3, 4), [-1]),
    (CylindricalSymGrid(3, (0, 3), 4, periodic_z=True), [1, -1]),
]


@pytest.mark.multiprocessing
def test_basic_mpi_methods():
    """test very basic methods"""
    mesh = GridMesh.from_grid(UnitGrid([4]))

    value = mesh.broadcast(mpi.rank)
    assert value == 0

    value = mesh.gather(mpi.rank)
    if mpi.is_main:
        assert value == list(range(mpi.size))
    else:
        assert value is None

    value = mesh.allgather(mpi.rank)
    assert value == list(range(mpi.size))


@pytest.mark.parametrize("grid, decomposition", GRIDS)
def test_generic_meshes(grid, decomposition):
    """test generic functions of the grid mesh"""
    mesh = GridMesh.from_grid(grid, decomposition)
    assert len(mesh) == mpi.size
    mesh.plot(action="close")


@pytest.mark.parametrize("decomp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_split_fields(decomp):
    """test splitting and recombining fields"""
    grid = UnitGrid([8, 8])
    mesh = GridMesh.from_grid(grid, decomp)

    field = ScalarField(grid)
    field._data_full = np.random.uniform(size=grid._shape_full)
    subfields = [mesh.extract_subfield(field, node_id) for node_id in range(len(mesh))]

    field_comb = mesh.combine_field_data([f.data for f in subfields])
    assert field == field_comb


@pytest.mark.multiprocessing
@pytest.mark.parametrize("decomp", [(-1,), (-1, 1), (1, -1)])
@pytest.mark.parametrize("dtype", [int, float, complex])
def test_split_fields_mpi(decomp, dtype):
    """test splitting and recombining fields using multiprocessing"""
    dim = len(decomp)
    grid = UnitGrid([8] * dim)
    mesh = GridMesh.from_grid(grid, decomp)

    field = ScalarField(grid, dtype=dtype)
    rng = np.random.default_rng(0)
    if dtype == int:
        field._data_full = rng.integers(0, 10, size=grid._shape_full)
    elif dtype == complex:
        field._data_full.real = rng.random(size=grid._shape_full)
        field._data_full.imag = rng.random(size=grid._shape_full)
    else:
        field._data_full = rng.random(size=grid._shape_full)

    # split without ghost cells
    subfield = mesh.split_field_mpi(field)
    field_data = mesh.combine_field_data_mpi(subfield.data)
    if mpi.is_main:
        assert subfield.label == field.label
        np.testing.assert_equal(field.data, field_data)
    else:
        assert field_data is None

    # split without ghost cells
    subfield_data = mesh.split_field_data_mpi(field.data, with_ghost_cells=False)
    np.testing.assert_equal(subfield.data, subfield_data)
    field_data = mesh.combine_field_data_mpi(subfield_data)
    if mpi.is_main:
        np.testing.assert_equal(field.data, field_data)
    else:
        assert field_data is None

    # split with ghost cells
    subfield_data = mesh.split_field_data_mpi(field._data_full, with_ghost_cells=True)
    np.testing.assert_equal(subfield._data_full, subfield_data)
    field_data = mesh.combine_field_data_mpi(
        subfield_data[(slice(1, -1),) * dim], with_ghost_cells=False
    )
    if mpi.is_main:
        np.testing.assert_equal(field.data, field_data)
    else:
        assert field_data is None


@pytest.mark.multiprocessing
@pytest.mark.parametrize("decomp", [(-1,), (-1, 1), (1, -1)])
def test_split_fieldcollections_mpi(decomp):
    """test splitting and recombining field collections using multiprocessing"""
    dim = len(decomp)
    grid = UnitGrid([8] * dim)
    mesh = GridMesh.from_grid(grid, decomp)

    rng = np.random.default_rng(0)
    sf = ScalarField.random_uniform(grid, rng=rng)
    vf = VectorField.random_uniform(grid, rng=rng)
    fc = FieldCollection([sf, vf], labels=["s", "v"])

    # split without ghost cells
    subfield = mesh.split_field_mpi(fc)
    sub_sf = mesh.split_field_mpi(sf)
    sub_vf = mesh.split_field_mpi(vf)
    sub_fc = FieldCollection([sub_sf, sub_vf], labels=["s", "v"])
    np.testing.assert_allclose(subfield.data, sub_fc.data)

    fc_data = mesh.combine_field_data_mpi(subfield.data)
    if mpi.is_main:
        assert subfield.labels == fc.labels
        np.testing.assert_equal(fc.data, fc_data)
    else:
        assert fc_data is None

    # split without ghost cells
    subfield_data = mesh.split_field_data_mpi(fc.data, with_ghost_cells=False)
    np.testing.assert_equal(subfield.data, subfield_data)
    fc_data = mesh.combine_field_data_mpi(subfield_data)
    if mpi.is_main:
        np.testing.assert_equal(fc.data, fc_data)
    else:
        assert fc_data is None

    # split with ghost cells
    subfield_data = mesh.split_field_data_mpi(fc._data_full, with_ghost_cells=True)
    np.testing.assert_equal(subfield._data_full, subfield_data)
    fc_data = mesh.combine_field_data_mpi(
        subfield_data[(...,) + (slice(1, -1),) * dim], with_ghost_cells=False
    )
    if mpi.is_main:
        np.testing.assert_equal(fc.data, fc_data)
    else:
        assert fc_data is None


@pytest.mark.multiprocessing
@pytest.mark.parametrize("bc", ["periodic", "value", "derivative", "curvature"])
def test_boundary_conditions_numpy(bc):
    """test setting boundary conditions using numpy"""
    grid = UnitGrid([8, 8], periodic=(bc == "periodic"))
    mesh = GridMesh.from_grid(grid)

    field = ScalarField.random_uniform(grid)

    # split without ghost cells
    f1 = mesh.split_field_mpi(field)
    f1.set_ghost_cells(bc)

    # split after setting ghost cells
    field.set_ghost_cells(bc)
    f2 = mesh.split_field_mpi(field)

    np.testing.assert_equal(f1._data_full[1:-1, :], f2._data_full[1:-1, :])
    np.testing.assert_equal(f1._data_full[:, 1:-1], f2._data_full[:, 1:-1])


@pytest.mark.multiprocessing
@pytest.mark.parametrize("bc", ["periodic", "value", "derivative", "curvature"])
def test_boundary_conditions_numba(bc):
    """test setting boundary conditions using numba"""
    grid = UnitGrid([8, 8], periodic=(bc == "periodic"))
    mesh = GridMesh.from_grid(grid)

    field = ScalarField.random_uniform(grid)

    # split without ghost cells
    f1 = mesh.split_field_mpi(field)
    bc1 = f1.grid.get_boundary_conditions(bc)
    bc1.make_ghost_cell_setter()(f1._data_full)

    # split after setting ghost cells
    bc2 = field.grid.get_boundary_conditions(bc)
    bc2.make_ghost_cell_setter()(field._data_full)
    f2 = mesh.split_field_mpi(field)

    np.testing.assert_equal(f1._data_full[1:-1, :], f2._data_full[1:-1, :])
    np.testing.assert_equal(f1._data_full[:, 1:-1], f2._data_full[:, 1:-1])


@pytest.mark.multiprocessing
@pytest.mark.parametrize("grid, decomposition", GRIDS)
def test_noncartesian_grids(grid, decomposition):
    """test whether we can deal with non-cartesian grids"""
    field = ScalarField.random_uniform(grid)
    eq = DiffusionPDE()

    args = {
        "state": field,
        "t_range": 1,
        "dt": 0.1,
        "backend": "numpy",
        "tracker": None,
    }
    res = eq.solve(method="explicit_mpi", decomposition=decomposition, **args)

    if mpi.is_main:
        # check results in the main process
        expect = eq.solve(method="explicit", **args)
        np.testing.assert_allclose(res.data, expect.data)


@pytest.mark.multiprocessing
@pytest.mark.parametrize("grid, decomposition", GRIDS)
@pytest.mark.parametrize("rank", [0, 2])
def test_integration_parallel(grid, decomposition, rank):
    """test integration of fields over grids"""
    mesh = GridMesh.from_grid(grid, decomposition=decomposition)
    if rank == 0:
        field = ScalarField(grid, 1)
        expected = grid.volume
    else:
        field = Tensor2Field(grid, 1)
        expected = np.full((grid.dim,) * 2, grid.volume)
    subfield = mesh.extract_subfield(field)

    # numpy version
    np.testing.assert_allclose(field.integral, expected)
    np.testing.assert_allclose(subfield.integral, expected)

    # numba version
    res = subfield.grid.make_integrator()(subfield.data)
    assert rank > 0 or np.isscalar(res)
    np.testing.assert_allclose(res, expected)
