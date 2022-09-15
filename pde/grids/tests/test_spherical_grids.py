"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, PolarSymGrid, ScalarField, SphericalSymGrid
from pde.grids.boundaries.local import NeumannBC


def test_polar_grid():
    """test simple polar grid"""
    grid = PolarSymGrid(4, 8)
    assert grid.dim == 2
    assert grid.num_cells == 8
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert not grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.5)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.5]))
    assert grid.volume == pytest.approx(np.pi * 4**2)
    assert grid.volume == pytest.approx(grid.integrate(1))

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(0.25, 3.75, 8))

    a = grid.make_operator("laplace", "auto_periodic_neumann")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    assert grid.contains_point(grid.get_random_point(coords="cartesian"))
    p = grid.get_random_point(boundary_distance=3.99, coords="cartesian")
    assert grid.contains_point(p)
    assert "laplace" in grid.operators


def test_polar_annulus():
    """test simple polar grid with a hole"""
    grid = PolarSymGrid((2, 4), 8)
    assert grid.dim == 2
    assert grid.num_cells == 8
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.25)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.25]))
    assert grid.volume == pytest.approx(np.pi * (4**2 - 2**2))
    assert grid.volume == pytest.approx(grid.integrate(1))
    assert grid.radius == (2, 4)

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(2.125, 3.875, 8))

    a = grid.make_operator("laplace", "auto_periodic_neumann")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    assert grid.contains_point(grid.get_random_point(coords="cartesian"))
    p = grid.get_random_point(boundary_distance=1.99, coords="cartesian")
    assert grid.contains_point(p)

    # test boundary points
    np.testing.assert_equal(grid._boundary_coordinates(0, False), np.array([2]))
    np.testing.assert_equal(grid._boundary_coordinates(0, True), np.array([4]))


def test_polar_to_cartesian():
    """test conversion of polar grid to Cartesian"""
    expr_pol = "(1 + r**2) ** -2"
    expr_cart = expr_pol.replace("r**2", "(x**2 + y**2)")

    grid_pol = PolarSymGrid(7, 16)
    pf_pol = ScalarField.from_expression(grid_pol, expression=expr_pol)

    grid_cart = CartesianGrid([[-4, 4], [-3.9, 4.1]], [16, 16])
    pf_cart1 = pf_pol.interpolate_to_grid(grid_cart)
    pf_cart2 = ScalarField.from_expression(grid_cart, expression=expr_cart)
    np.testing.assert_allclose(pf_cart1.data, pf_cart2.data, atol=0.1)


def test_spherical_grid():
    """test simple spherical grid"""
    grid = SphericalSymGrid(4, 8)
    assert grid.dim == 3
    assert grid.num_cells == 8
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert not grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.5)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.5]))
    assert grid.volume == pytest.approx(4 / 3 * np.pi * 4**3)
    assert grid.volume == pytest.approx(grid.integrate(1))

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(0.25, 3.75, 8))

    a = grid.make_operator("laplace", "auto_periodic_neumann")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    assert grid.contains_point(grid.get_random_point(coords="cartesian"))
    p = grid.get_random_point(boundary_distance=3.99, coords="cartesian")
    assert grid.contains_point(p)
    assert "laplace" in grid.operators


def test_spherical_annulus():
    """test simple spherical grid with a hole"""
    grid = SphericalSymGrid((2, 4), 8)
    assert grid.dim == 3
    assert grid.num_cells == 8
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.25)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.25]))
    assert grid.volume == pytest.approx(4 / 3 * np.pi * (4**3 - 2**3))
    assert grid.volume == pytest.approx(grid.integrate(1))
    assert grid.radius == (2, 4)

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(2.125, 3.875, 8))

    a = grid.make_operator("laplace", "auto_periodic_neumann")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    assert grid.contains_point(grid.get_random_point(coords="cartesian"))
    p = grid.get_random_point(boundary_distance=1.99, coords="cartesian")
    assert grid.contains_point(p)

    # test boundary points
    np.testing.assert_equal(grid._boundary_coordinates(0, False), np.array([2]))
    np.testing.assert_equal(grid._boundary_coordinates(0, True), np.array([4]))


def test_spherical_to_cartesian():
    """test conversion of spherical grid to cartesian"""
    expr_sph = "1. / (1 + r**2)"
    expr_cart = expr_sph.replace("r**2", "(x**2 + y**2 + z**2)")

    grid_sph = SphericalSymGrid(7, 16)
    pf_sph = ScalarField.from_expression(grid_sph, expression=expr_sph)

    grid_cart = CartesianGrid([[-4, 4], [-3.9, 4.1], [-4.1, 3.9]], [16] * 3)
    pf_cart1 = pf_sph.interpolate_to_grid(grid_cart)
    pf_cart2 = ScalarField.from_expression(grid_cart, expression=expr_cart)
    np.testing.assert_allclose(pf_cart1.data, pf_cart2.data, atol=0.1)


@pytest.mark.parametrize("grid_class", [PolarSymGrid, SphericalSymGrid])
def test_setting_boundary_conditions(grid_class):
    """test setting some boundary conditions"""
    grid = grid_class([0, 1], 3)
    b_inner = NeumannBC(grid, 0, upper=False)

    assert grid.get_boundary_conditions("auto_periodic_neumann")[0].low == b_inner
    assert grid.get_boundary_conditions(["derivative", {"value": 2}])[0].low == b_inner
    bcs = grid.get_boundary_conditions(["value", "value"])
    assert bcs[0].low != b_inner

    grid = grid_class([1, 2], 3)
    bcs = grid.get_boundary_conditions(["value", "value"])
    assert bcs[0].low != b_inner
