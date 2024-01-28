"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import random
from functools import partial

import numpy as np
import pytest

from pde import CartesianGrid, UnitGrid
from pde.grids.boundaries import Boundaries, PeriodicityError
from pde.grids.operators.common import make_derivative


def _get_cartesian_grid(dim=2, periodic=True):
    """return a random Cartesian grid of given dimension"""
    rng = np.random.default_rng(0)
    bounds = [[0, 1 + rng.random()] for _ in range(dim)]
    shape = rng.integers(32, 64, size=dim)
    return CartesianGrid(bounds, shape, periodic=periodic)


def test_degenerated_grid():
    """test degenerated grids"""
    with pytest.raises(ValueError):
        UnitGrid([])
    with pytest.raises(ValueError):
        CartesianGrid([], 1)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_generic_cartesian_grid(dim, rng):
    """test generic cartesian grid functions"""
    periodic = random.choices([True, False], k=dim)
    shape = rng.integers(2, 8, size=dim)
    a = rng.random(dim)
    b = a + rng.random(dim)

    cases = [
        UnitGrid(shape, periodic=periodic),
        CartesianGrid(np.c_[a, b], shape, periodic=periodic),
    ]
    for grid in cases:
        assert grid.dim == dim
        assert grid.num_cells == np.prod(shape)
        dim_axes = len(grid.axes) + len(grid.axes_symmetric)
        assert dim_axes == dim
        vol = np.prod(grid.discretization) * np.prod(shape)
        assert grid.volume == pytest.approx(vol)
        assert grid.uniform_cell_volumes

        assert grid.contains_point(grid.get_random_point(coords="grid", rng=rng))
        w = 0.499 * (b - a).min()
        p = grid.get_random_point(boundary_distance=w, coords="grid", rng=rng)
        assert grid.contains_point(p)
        assert "laplace" in grid.operators


@pytest.mark.parametrize("periodic", [True, False])
def test_unit_grid_1d(periodic, rng):
    """test 1D grids"""
    grid = UnitGrid(4, periodic=periodic)
    assert grid.dim == 1
    assert grid.numba_type == "f8[:]"
    assert grid.volume == 4
    np.testing.assert_array_equal(grid.discretization, np.ones(1))

    grid = UnitGrid(8, periodic=periodic)
    assert grid.dim == 1
    assert grid.volume == 8

    norm_numba = grid.make_normalize_point_compiled(reflect=False)

    def norm_numba_wrap(x):
        y = np.array([x])
        norm_numba(y)
        return y

    for normalize in [partial(grid.normalize_point, reflect=False), norm_numba_wrap]:
        if periodic:
            np.testing.assert_allclose(normalize(-1e-10), 8 - 1e-10)
            np.testing.assert_allclose(normalize(1e-10), 1e-10)
            np.testing.assert_allclose(normalize(8 - 1e-10), 8 - 1e-10)
            np.testing.assert_allclose(normalize(8 + 1e-10), 1e-10)
        else:
            for x in [-1e-10, 1e-10, 8 - 1e-10, 8 + 1e-10]:
                np.testing.assert_allclose(normalize(x), x)

    grid = UnitGrid(8, periodic=periodic)

    # test boundary points
    np.testing.assert_equal(grid._boundary_coordinates(0, False), np.array([0]))
    np.testing.assert_equal(grid._boundary_coordinates(0, True), np.array([8]))


def test_unit_grid_2d(rng):
    """test 2D grids"""
    # test special case
    grid = UnitGrid([4, 4], periodic=True)
    assert grid.dim == 2
    assert grid.numba_type == "f8[:, :]"
    assert grid.volume == 16
    np.testing.assert_array_equal(grid.discretization, np.ones(2))
    assert grid.get_image_data(np.zeros(grid.shape))["extent"] == [0, 4, 0, 4]

    periodic = random.choices([True, False], k=2)
    grid = UnitGrid([4, 4], periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 16

    grid = UnitGrid([4, 8], periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 32

    # test boundary points
    np.testing.assert_equal(
        grid._boundary_coordinates(0, False),
        np.c_[np.full(8, 0), np.linspace(0.5, 7.5, 8)],
    )
    np.testing.assert_equal(
        grid._boundary_coordinates(0, True),
        np.c_[np.full(8, 4), np.linspace(0.5, 7.5, 8)],
    )
    np.testing.assert_equal(
        grid._boundary_coordinates(1, False),
        np.c_[np.linspace(0.5, 3.5, 4), np.full(4, 0)],
    )
    np.testing.assert_equal(
        grid._boundary_coordinates(1, True),
        np.c_[np.linspace(0.5, 3.5, 4), np.full(4, 8)],
    )


def test_unit_grid_3d(rng):
    """test 3D grids"""
    grid = UnitGrid([4, 4, 4])
    assert grid.dim == 3
    assert grid.numba_type == "f8[:, :, :]"
    assert grid.volume == 64
    np.testing.assert_array_equal(grid.discretization, np.ones(3))
    assert grid.get_image_data(np.zeros(grid.shape))["extent"] == [0, 4, 0, 4]

    periodic = random.choices([True, False], k=3)
    grid = UnitGrid([4, 6, 8], periodic=periodic)
    assert grid.dim == 3
    assert grid.volume == 192

    grid = UnitGrid([4, 4, 4], periodic=True)
    assert grid.dim == 3
    assert grid.volume == 64

    # test boundary points
    for bndry in grid._iter_boundaries():
        assert grid._boundary_coordinates(*bndry).shape == (4, 4, 3)


def test_rect_grid_1d(rng):
    """test 1D grids"""
    grid = CartesianGrid([32], 16, periodic=False)
    assert grid.dim == 1
    assert grid.volume == 32
    assert grid.typical_discretization == 2
    np.testing.assert_array_equal(grid.discretization, np.full(1, 2))

    grid = CartesianGrid([[-16, 16]], 8, periodic=True)
    assert grid.cuboid.pos == [-16]
    assert grid.shape == (8,)
    assert grid.dim == 1
    assert grid.volume == 32
    assert grid.typical_discretization == 4

    np.testing.assert_allclose(grid.normalize_point(-16 - 1e-10), 16 - 1e-10)
    np.testing.assert_allclose(grid.normalize_point(-16 + 1e-10), -16 + 1e-10)
    np.testing.assert_allclose(grid.normalize_point(16 - 1e-10), 16 - 1e-10)
    np.testing.assert_allclose(grid.normalize_point(16 + 1e-10), -16 + 1e-10)


def test_rect_grid_2d(rng):
    """test 2D grids"""
    grid = CartesianGrid([[2], [2]], 4, periodic=True)
    assert grid.get_image_data(np.zeros(grid.shape))["extent"] == [0, 2, 0, 2]

    periodic = random.choices([True, False], k=2)
    grid = CartesianGrid([[4], [4]], 4, periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 16
    np.testing.assert_array_equal(grid.discretization, np.ones(2))
    assert grid.typical_discretization == 1

    grid = CartesianGrid([[-2, 2], [-2, 2]], [4, 8], periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 16
    assert grid.typical_discretization == 0.75


def test_rect_grid_3d(rng):
    """test 3D grids"""
    grid = CartesianGrid([4, 4, 4], 4)
    assert grid.dim == 3
    assert grid.volume == 64
    assert grid.typical_discretization == 1
    np.testing.assert_array_equal(grid.discretization, np.ones(3))

    bounds = [[-2, 2], [-2, 2], [-2, 2]]
    grid = CartesianGrid(bounds, [4, 6, 8])
    assert grid.dim == 3
    np.testing.assert_allclose(grid.axes_bounds, bounds)
    assert grid.volume == 64
    assert grid.typical_discretization == pytest.approx(0.7222222222222)


@pytest.mark.parametrize("periodic", [True, False])
def test_unit_rect_grid(periodic, rng):
    """test whether the rectangular grid behaves like a unit grid in special cases"""
    dim = random.randrange(1, 4)
    shape = rng.integers(2, 10, size=dim)
    g1 = UnitGrid(shape, periodic=periodic)
    g2 = CartesianGrid(np.c_[np.zeros(dim), shape], shape, periodic=periodic)
    volume = np.prod(shape)
    for g in [g1, g2]:
        assert g.volume == pytest.approx(volume)
        assert g.integrate(1) == pytest.approx(volume)
        assert g.make_integrator()(np.ones(shape)) == pytest.approx(volume)

    assert g1.dim == g2.dim == dim
    np.testing.assert_array_equal(g1.shape, g2.shape)
    assert g1.typical_discretization == pytest.approx(g2.typical_discretization)

    for _ in range(10):
        p1, p2 = rng.normal(scale=10, size=(2, dim))
        assert g1.distance(p1, p2) == pytest.approx(g2.distance(p1, p2))


def test_conversion_unit_rect_grid(rng):
    """test the conversion from unit to rectangular grid"""
    dim = random.randrange(1, 4)
    shape = rng.integers(2, 10, size=dim)
    periodic = random.choices([True, False], k=dim)
    g1 = UnitGrid(shape, periodic=periodic)
    g2 = g1.to_cartesian()

    assert g1.shape == g2.shape
    assert g1.cuboid == g2.cuboid
    assert g1.periodic == g2.periodic


def test_setting_boundary_conditions():
    """test setting some boundary conditions"""
    grid = UnitGrid([3, 3], periodic=[True, False])
    for bc in [
        grid.get_boundary_conditions("auto_periodic_neumann"),
        grid.get_boundary_conditions(["auto_periodic_neumann", "derivative"]),
    ]:
        assert isinstance(bc, Boundaries)

    for bc in ["periodic", "value"]:
        with pytest.raises(PeriodicityError):
            grid.get_boundary_conditions(bc)

    grid = UnitGrid([2], periodic=True)
    with pytest.raises(PeriodicityError):
        grid.get_boundary_conditions("derivative")

    grid = UnitGrid([2], periodic=False)
    with pytest.raises(PeriodicityError):
        grid.get_boundary_conditions("periodic")


def test_setting_domain_rect():
    """test various versions of settings bcs for Cartesian grids"""
    grid = UnitGrid([2, 2])
    grid.get_boundary_conditions(["derivative", "derivative"])

    # wrong number of conditions
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(["derivative"])
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(["derivative"] * 3)

    grid = UnitGrid([2, 2], periodic=[True, False])
    grid.get_boundary_conditions("auto_periodic_neumann")
    grid.get_boundary_conditions(["periodic", "derivative"])

    # incompatible conditions
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions("periodic")
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions("derivative")
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions(["derivative", "periodic"])


@pytest.mark.parametrize("reflect", [True, False])
def test_normalize_point(reflect):
    """test normalize_point method for Cartesian Grids"""
    grid = CartesianGrid([[1, 3]], [1], periodic=False)

    norm_numba = grid.make_normalize_point_compiled(reflect=reflect)

    def norm_numba_wrap(x):
        y = np.array([x])
        norm_numba(y)
        return y

    if reflect:
        values = [(-2, 2), (0, 2), (1, 1), (2, 2), (3, 3), (4, 2), (5, 1), (6, 2)]
    else:
        values = [(-2, -2), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

    for norm in [norm_numba_wrap, partial(grid.normalize_point, reflect=reflect)]:
        for x, y in values:
            assert norm(x) == pytest.approx(y), (norm, x)


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_generic_operators(method, rng):
    """test the `d_dx` version of the operator"""
    grid = CartesianGrid([[1, 3]], 5, periodic=True)
    bcs = grid.get_boundary_conditions("periodic")
    data = rng.uniform(0, 1, size=5)
    data_full = np.empty(7)
    data_full[1:-1] = data
    bcs.set_ghost_cells(data_full)

    op1 = make_derivative(grid, axis=0, method=method)
    expect = np.empty(5)
    op1(data_full, expect)
    op2 = grid.make_operator(f"d_dx_{method}", bc=bcs)
    np.testing.assert_allclose(expect, op2(data))

    if method == "central":
        op3 = grid.make_operator("gradient", bc="periodic")
        np.testing.assert_allclose(expect, op3(data)[0])


def test_boundary_coordinates():
    """test _boundary_coordinates method"""
    grid = UnitGrid([2, 2])

    c = grid._boundary_coordinates(axis=0, upper=False)
    np.testing.assert_allclose(c, [[0.0, 0.5], [0.0, 1.5]])
    c = grid._boundary_coordinates(axis=0, upper=False, offset=0.5)
    np.testing.assert_allclose(c, [[0.5, 0.5], [0.5, 1.5]])

    c = grid._boundary_coordinates(axis=0, upper=True)
    np.testing.assert_allclose(c, [[2.0, 0.5], [2.0, 1.5]])
    c = grid._boundary_coordinates(axis=0, upper=True, offset=0.5)
    np.testing.assert_allclose(c, [[1.5, 0.5], [1.5, 1.5]])
