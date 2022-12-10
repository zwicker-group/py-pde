"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools
from copy import copy, deepcopy

import numpy as np
import pytest

from pde import grids
from pde.grids.base import GridBase, discretize_interval, registered_operators
from pde.tools.misc import skipUnlessModule


def iter_grids():
    """generator providing some test grids"""
    for periodic in [True, False]:
        yield grids.UnitGrid([3], periodic=periodic)
        yield grids.UnitGrid([3, 3, 3], periodic=periodic)
        yield grids.CartesianGrid([[-1, 2], [0, 3]], [5, 7], periodic=periodic)
        yield grids.CylindricalSymGrid(3, [-1, 2], [7, 8], periodic_z=periodic)
    yield grids.PolarSymGrid(3, 4)
    yield grids.SphericalSymGrid(3, 4)


def test_discretize():
    """test the discretize function"""
    x_min = np.random.uniform(0, 1)
    x_max = np.random.uniform(2, 3)
    num = np.random.randint(5, 8)
    x, dx = discretize_interval(x_min, x_max, num)
    assert dx == pytest.approx((x_max - x_min) / num)
    x_expect = np.linspace(x_min + dx / 2, x_max - dx / 2, num)
    np.testing.assert_allclose(x, x_expect)


@pytest.mark.parametrize("grid", iter_grids())
def test_serialization(grid):
    """test whether grid can be serialized and copied"""
    g = GridBase.from_state(grid.state_serialized)
    assert grid == g
    assert grid._cache_hash() == g._cache_hash()

    for g in (grid.copy(), copy(grid), deepcopy(grid)):
        assert grid == g
        assert grid is not g


def test_iter_mirror_points():
    """test iterating mirror points in grids"""
    grid_cart = grids.UnitGrid([2, 2], periodic=[True, False])
    grid_cyl = grids.CylindricalSymGrid(2, (0, 2), (2, 2), periodic_z=False)
    grid_sph = grids.SphericalSymGrid(2, 2)
    assert grid_cart._cache_hash() != grid_cyl._cache_hash() != grid_sph._cache_hash()

    for with_, only_periodic in itertools.product([False, True], repeat=2):
        num_expect = 2 if only_periodic else 8
        num_expect += 1 if with_ else 0
        ps = grid_cart.iter_mirror_points([1, 1], with_, only_periodic)
        assert len(list(ps)) == num_expect

        num_expect = 0 if only_periodic else 2
        num_expect += 1 if with_ else 0
        ps = grid_cyl.iter_mirror_points([0, 0, 1], with_, only_periodic)
        assert len(list(ps)) == num_expect

        num_expect = 1 if with_ else 0
        ps = grid_sph.iter_mirror_points([0, 0, 0], with_, only_periodic)
        assert len(list(ps)) == num_expect


@pytest.mark.parametrize("grid", iter_grids())
def test_coordinate_conversion(grid):
    """test the conversion between cells and points"""
    p_empty = np.zeros((0, grid.dim))
    c_empty = np.zeros((0, grid.num_axes))

    p = grid.get_random_point(coords="grid")
    for coords in ["cartesian", "cell", "grid"]:
        # test empty conversion
        assert grid.transform(p_empty, "cartesian", coords).size == 0
        assert grid.transform(c_empty, "grid", coords).size == 0
        assert grid.transform(c_empty, "cell", coords).size == 0

        # test full conversion
        p1 = grid.transform(p, "grid", coords)
        for target in ["cartesian", "grid"]:
            p2 = grid.transform(p1, coords, target)
            p3 = grid.transform(p2, target, coords)
            np.testing.assert_allclose(p1, p3, err_msg=f"{coords} -> {target}")


@pytest.mark.parametrize("grid", iter_grids())
def test_integration_serial(grid):
    """test integration of fields"""
    arr = np.random.randn(*grid.shape)
    res = grid.make_integrator()(arr)
    assert np.isscalar(res)
    assert res == pytest.approx(grid.integrate(arr))
    if grid.num_axes == 1:
        assert res == pytest.approx(grid.integrate(arr, axes=0))
    else:
        assert res == pytest.approx(grid.integrate(arr, axes=range(grid.num_axes)))


@skipUnlessModule("matplotlib")
def test_grid_plotting():
    """test plotting of grids"""
    grids.UnitGrid([4]).plot()
    grids.UnitGrid([4, 4]).plot()

    with pytest.raises(NotImplementedError):
        grids.UnitGrid([4, 4, 4]).plot()

    grids.PolarSymGrid(4, 8).plot()
    grids.PolarSymGrid((2, 4), 8).plot()


@pytest.mark.parametrize("grid", iter_grids())
def test_operators(grid):
    """test operator mechanism"""

    def make_op(state):
        return lambda state: state

    assert "laplace" in grid.operators
    with pytest.raises(ValueError):
        grid.make_operator("not_existent", "auto_periodic_neumann")
    grid.register_operator("noop", make_op)
    assert "noop" in grid.operators
    del grid._operators["noop"]  # reset original state


def test_registered_operators():
    """test the registered_operators function"""
    for grid_name, ops in registered_operators().items():
        grid_class_ops = getattr(grids, grid_name).operators
        assert all(op in grid_class_ops for op in ops)
