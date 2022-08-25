"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CylindricalSymGrid, DiffusionPDE, PolarSymGrid, ScalarField, UnitGrid
from pde.grids.mesh import GridMesh
from pde.tools import mpi


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
@pytest.mark.parametrize("decomp", [(-1, 1), (1, -1)])
def test_split_fields_mpi(decomp):
    """test splitting and recombining fields using multiprocessing"""
    grid = UnitGrid([8, 8])
    mesh = GridMesh.from_grid(grid, decomp)

    field = ScalarField(grid)
    field._data_full = np.random.uniform(size=grid._shape_full)

    # split without ghost cells
    subfield = mesh.split_field_mpi(field)
    field_data = mesh.combine_field_data_mpi(subfield.data)
    if mesh.current_node == 0:
        np.testing.assert_equal(field.data, field_data)

    # split without ghost cells
    subfield_data = mesh.split_field_data_mpi(field.data, with_ghost_cells=False)
    field_data = mesh.combine_field_data_mpi(subfield_data)
    if mesh.current_node == 0:
        np.testing.assert_equal(field.data, field_data)

    # split with ghost cells
    subfield_data = mesh.split_field_data_mpi(field._data_full, with_ghost_cells=True)
    field_data = mesh.combine_field_data_mpi(subfield_data[1:-1, 1:-1])
    if mesh.current_node == 0:
        np.testing.assert_equal(field.data, field_data)


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
@pytest.mark.parametrize(
    "grid", [PolarSymGrid(3, 4), CylindricalSymGrid(3, (0, 3), 4, periodic_z=True)]
)
def test_noncartesian_grids(grid):
    """test whether we can deal with non-cartesian grids"""
    if isinstance(grid, CylindricalSymGrid):
        decomposition = [1, -1]
    else:
        decomposition = [-1]

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
