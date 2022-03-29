"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, CylindricalSymGrid, ScalarField
from pde.grids.boundaries.local import NeumannBC


def test_cylindrical_grid():
    """test simple cylindrical grid"""
    for periodic in [True, False]:
        grid = CylindricalSymGrid(4, (-1, 2), (8, 9), periodic_z=periodic)

        assert grid.dim == 3
        assert grid.numba_type == "f8[:, :]"
        assert grid.shape == (8, 9)
        assert grid.length == pytest.approx(3)
        assert grid.discretization[0] == pytest.approx(0.5)
        assert grid.discretization[1] == pytest.approx(1 / 3)
        np.testing.assert_array_equal(grid.discretization, np.array([0.5, 1 / 3]))
        assert not grid.uniform_cell_volumes
        assert grid.volume == pytest.approx(np.pi * 4**2 * 3)
        assert grid.volume == pytest.approx(grid.integrate(1))

        rs, zs = grid.axes_coords
        np.testing.assert_allclose(rs, np.linspace(0.25, 3.75, 8))
        np.testing.assert_allclose(zs, np.linspace(-1 + 1 / 6, 2 - 1 / 6, 9))

        assert grid.contains_point(grid.get_random_point(coords="cartesian"))
        ps = [grid.get_random_point(coords="cartesian") for _ in range(2)]
        assert all(grid.contains_point(ps))
        ps = grid.get_random_point(coords="cartesian", boundary_distance=1.49)
        assert grid.contains_point(ps)
        assert "laplace" in grid.operators


def test_cylindrical_to_cartesian():
    """test conversion of cylindrical grid to Cartesian"""
    expr_cyl = "cos(z / 2) / (1 + r**2)"
    expr_cart = expr_cyl.replace("r**2", "(x**2 + y**2)")

    z_range = (-np.pi, 2 * np.pi)
    grid_cyl = CylindricalSymGrid(10, z_range, (16, 33))
    pf_cyl = ScalarField.from_expression(grid_cyl, expression=expr_cyl)

    grid_cart = CartesianGrid([[-7, 7], [-6, 7], z_range], [16, 16, 16])
    pf_cart1 = pf_cyl.interpolate_to_grid(grid_cart)
    pf_cart2 = ScalarField.from_expression(grid_cart, expression=expr_cart)
    np.testing.assert_allclose(pf_cart1.data, pf_cart2.data, atol=0.1)


def test_setting_domain_cylindrical():
    """test various versions of settings bcs for cylindrical grids"""
    grid = CylindricalSymGrid(1, [0, 1], [2, 2], periodic_z=False)
    grid.get_boundary_conditions("auto_periodic_neumann")
    grid.get_boundary_conditions(["derivative", "derivative"])
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(["derivative"])
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(["derivative"] * 3)
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions(["derivative", "periodic"])

    grid = CylindricalSymGrid(1, [0, 1], [2, 2], periodic_z=True)
    grid.get_boundary_conditions("auto_periodic_neumann")
    grid.get_boundary_conditions(["derivative", "periodic"])
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions(["derivative", "derivative"])


@pytest.mark.parametrize("periodic", [True, False])
def test_polar_conversion(periodic):
    """test conversion to polar coordinates"""
    grid = CylindricalSymGrid(1, [-1, 1], [5, 5], periodic_z=periodic)
    dists = grid.polar_coordinates_real([0, 0, 0])
    assert np.all(0.09 <= dists)
    assert np.any(dists < 0.11)
    assert np.all(dists <= np.sqrt(2))
    assert np.any(dists > 0.8 * np.sqrt(2))


def test_setting_boundary_conditions():
    """test setting some boundary conditions"""
    grid = CylindricalSymGrid(1, [0, 1], 3)
    b_inner = NeumannBC(grid, 0, upper=False)

    assert grid.get_boundary_conditions("auto_periodic_neumann")[0].low == b_inner
    assert grid.get_boundary_conditions({"value": 2})[0].low != b_inner
