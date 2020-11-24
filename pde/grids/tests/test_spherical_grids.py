"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde import CartesianGrid, PolarGrid, ScalarField, SphericalGrid
from pde.grids.boundaries.local import NeumannBC


@pytest.mark.parametrize("grid_class", [PolarGrid, SphericalGrid])
def test_spherical_base_bcs(grid_class):
    """ test setting boundary conditions on spherical grids """
    grid = grid_class(2, 3)

    domain1 = grid.get_boundary_conditions(["derivative", {"type": "value"}])
    domain2 = grid.get_boundary_conditions({"type": "value"})
    assert domain1 == domain2

    # test boundary conditions for simulations with holes
    grid = grid_class((1, 2), 3)
    grid.get_boundary_conditions(["derivative", {"type": "value"}])
    domain1 = grid.get_boundary_conditions({"type": "value"})
    domain2 = grid.get_boundary_conditions(["value", "value"])
    assert domain1 == domain2


def test_polar_grid():
    """ test simple polar grid """
    grid = PolarGrid(4, 8)
    assert grid.dim == 2
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert not grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.5)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.5]))
    assert grid.volume == pytest.approx(np.pi * 4 ** 2)
    assert grid.volume == pytest.approx(grid.integrate(1))

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(0.25, 3.75, 8))

    a = grid.get_operator("laplace", "natural")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    # random points
    c = np.random.randint(8, size=(6, 1))
    p = grid.cell_to_point(c)
    np.testing.assert_array_equal(c, grid.point_to_cell(p))

    assert grid.contains_point(grid.get_random_point())
    assert grid.contains_point(grid.get_random_point(3.99))
    assert "laplace" in grid.operators


def test_polar_annulus():
    """ test simple polar grid with a hole """
    grid = PolarGrid((2, 4), 8)
    assert grid.dim == 2
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.25)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.25]))
    assert grid.volume == pytest.approx(np.pi * (4 ** 2 - 2 ** 2))
    assert grid.volume == pytest.approx(grid.integrate(1))
    assert grid.radius == (2, 4)

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(2.125, 3.875, 8))

    a = grid.get_operator("laplace", "natural")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    # random points
    c = np.random.randint(8, size=(6, 1))
    p = grid.cell_to_point(c)
    np.testing.assert_array_equal(c, grid.point_to_cell(p))

    assert grid.contains_point(grid.get_random_point())
    assert grid.contains_point(grid.get_random_point(1.99))

    # test boundary points
    np.testing.assert_equal(grid._boundary_coordinates(0, False), np.array([2]))
    np.testing.assert_equal(grid._boundary_coordinates(0, True), np.array([4]))


def test_polar_to_cartesian():
    """ test conversion of polar grid to Cartesian """
    expr_pol = "1 / (1 + r**2)"
    expr_cart = expr_pol.replace("r**2", "(x**2 + y**2)")

    grid_pol = PolarGrid(7, 16)
    pf_pol = ScalarField.from_expression(grid_pol, expression=expr_pol)

    grid_cart = CartesianGrid([[-4, 4], [-3.9, 4.1]], [16, 16])
    pf_cart1 = pf_pol.interpolate_to_grid(grid_cart)
    pf_cart2 = ScalarField.from_expression(grid_cart, expression=expr_cart)
    np.testing.assert_allclose(pf_cart1.data, pf_cart2.data, atol=0.1)


def test_spherical_grid():
    """ test simple spherical grid """
    grid = SphericalGrid(4, 8)
    assert grid.dim == 3
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert not grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.5)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.5]))
    assert grid.volume == pytest.approx(4 / 3 * np.pi * 4 ** 3)
    assert grid.volume == pytest.approx(grid.integrate(1))

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(0.25, 3.75, 8))

    a = grid.get_operator("laplace", "natural")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    # random points
    c = np.random.randint(8, size=(6, 1))
    p = grid.cell_to_point(c)
    np.testing.assert_array_equal(c, grid.point_to_cell(p))

    assert grid.contains_point(grid.get_random_point())
    assert grid.contains_point(grid.get_random_point(3.99))
    assert "laplace" in grid.operators


def test_spherical_annulus():
    """ test simple spherical grid with a hole """
    grid = SphericalGrid((2, 4), 8)
    assert grid.dim == 3
    assert grid.numba_type == "f8[:]"
    assert grid.shape == (8,)
    assert grid.has_hole
    assert grid.discretization[0] == pytest.approx(0.25)
    assert not grid.uniform_cell_volumes
    np.testing.assert_array_equal(grid.discretization, np.array([0.25]))
    assert grid.volume == pytest.approx(4 / 3 * np.pi * (4 ** 3 - 2 ** 3))
    assert grid.volume == pytest.approx(grid.integrate(1))
    assert grid.radius == (2, 4)

    np.testing.assert_allclose(grid.axes_coords[0], np.linspace(2.125, 3.875, 8))

    a = grid.get_operator("laplace", "natural")(np.random.random(8))
    assert a.shape == (8,)
    assert np.all(np.isfinite(a))

    # random points
    c = np.random.randint(8, size=(6, 1))
    p = grid.cell_to_point(c)
    assert all(grid.contains_point(r) for r in p)
    np.testing.assert_array_equal(c, grid.point_to_cell(p))

    assert grid.contains_point(grid.get_random_point())
    assert grid.contains_point(grid.get_random_point(1.99))

    # test boundary points
    np.testing.assert_equal(grid._boundary_coordinates(0, False), np.array([2]))
    np.testing.assert_equal(grid._boundary_coordinates(0, True), np.array([4]))


def test_spherical_to_cartesian():
    """ test conversion of spherical grid to cartesian """
    expr_sph = "1 / (1 + r**2)"
    expr_cart = expr_sph.replace("r**2", "(x**2 + y**2 + z**2)")

    grid_sph = SphericalGrid(7, 16)
    pf_sph = ScalarField.from_expression(grid_sph, expression=expr_sph)

    grid_cart = CartesianGrid([[-4, 4], [-3.9, 4.1], [-4.1, 3.9]], [16] * 3)
    pf_cart1 = pf_sph.interpolate_to_grid(grid_cart)
    pf_cart2 = ScalarField.from_expression(grid_cart, expression=expr_cart)
    np.testing.assert_allclose(pf_cart1.data, pf_cart2.data, atol=0.1)


@pytest.mark.parametrize("grid_class", [PolarGrid, SphericalGrid])
def test_setting_boundary_conditions(grid_class):
    """ test setting some boundary conditions """
    grid = grid_class([0, 1], 3)
    b_inner = NeumannBC(grid, 0, upper=False)

    assert grid.get_boundary_conditions("natural")[0].low == b_inner
    assert grid.get_boundary_conditions({"value": 2})[0].low == b_inner
    bcs = grid.get_boundary_conditions(["value", "value"])
    assert bcs[0].low != b_inner

    grid = grid_class([1, 2], 3)
    bcs = grid.get_boundary_conditions(["value", "value"])
    assert bcs[0].low != b_inner
