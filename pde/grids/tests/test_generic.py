'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import itertools
import random

import numpy as np
import pytest

from ... import grids
from ..base import discretize_interval, GridBase
from ...tools.misc import skipUnlessModule



def iter_grids():
    """ generate some test grids """
    yield grids.UnitGrid([2, 2], periodic=[True, False])
    yield grids.CartesianGrid([[0, 1]], [2], periodic=[False])
    yield grids.CylindricalGrid(2, (0, 2), (2, 2), periodic_z=True)
    yield grids.SphericalGrid(2, 2)
    yield grids.PolarGrid(2, 2)



def test_discretize():
    """ test the discretize function """
    x_min = np.random.uniform(0, 1)
    x_max = np.random.uniform(2, 3)
    num = np.random.randint(5, 8)
    x, dx = discretize_interval(x_min, x_max, num)
    assert dx == pytest.approx((x_max - x_min) / num)
    x_expect = np.linspace(x_min + dx / 2, x_max - dx / 2, num)
    np.testing.assert_allclose(x, x_expect)



def test_generic_grid():
    """ test generic grid functions """
    for dim in (1, 2, 3):
        periodic = random.choices([True, False], k=dim)
        shape = np.random.randint(2, 8, size=dim)
        a = np.random.random(dim)
        b = a + np.random.random(dim)
        
        cases = [grids.UnitGrid(shape, periodic=periodic),
                 grids.CartesianGrid(np.c_[a, b], shape, periodic=periodic)]
        for grid in cases:
            assert grid.dim == dim
            dim_axes = len(grid.axes) + len(grid.axes_symmetric)
            assert dim_axes == dim
            vol = grid.cell_volume_data * np.prod(shape)
            assert grid.volume == pytest.approx(vol)
            
            # random points
            points = [[np.random.uniform(a[i], b[i]) for i in range(dim)]
                      for _ in range(10)]
            c = grid.point_to_cell(points)
            p = grid.cell_to_point(c)
            np.testing.assert_array_equal(c, grid.point_to_cell(p))
            
            assert grid.contains_point(grid.get_random_point())
            w = 0.499 * (b - a).min()
            assert grid.contains_point(grid.get_random_point(w))



def test_serialization():
    """ test whether grid can be serialized """
    def iter_grids():
        """ helper function iterating over different grids """
        for periodic in [True, False]:
            yield grids.UnitGrid([3, 4], periodic=periodic)
            yield grids.CartesianGrid([[0, 1], [-2, 3]], [4, 5],
                                      periodic=periodic)
            yield grids.CylindricalGrid(3, [-1, 2], [5, 7],
                                        periodic_z=periodic)
        yield grids.SphericalGrid(4, 6)
        yield grids.PolarGrid(4, 5)
            
    for grid in iter_grids():
        g = GridBase.from_state(grid.state_serialized)
        assert grid == g
            
        
        
def test_iter_mirror_points():
    """ test iterating mirror points in grids """
    grid_cart = grids.UnitGrid([2, 2], periodic=[True, False])
    grid_cyl = grids.CylindricalGrid(2, (0, 2), (2, 2), periodic_z=False)
    grid_sph = grids.SphericalGrid(2, 2)
    
    for with_, only_periodic in itertools.product([False, True], repeat=2):
        num_expect = 2 if only_periodic else 8
        num_expect += 1 if with_ else 0
        ps = grid_cart.iter_mirror_points([1, 1], with_, only_periodic)
        assert len(list(ps)) == num_expect
        
        num_expect = 0 if only_periodic else 2
        num_expect += 1 if with_ else 0
        ps = grid_cyl.iter_mirror_points([0, 0, 1], with_,
                                         only_periodic)
        assert len(list(ps)) == num_expect
        
        num_expect = 1 if with_ else 0
        ps = grid_sph.iter_mirror_points([0, 0, 0], with_,
                                         only_periodic)
        assert len(list(ps)) == num_expect
        
        
        
def test_cell_to_point_conversion():
    """ test the conversion between cells and points """
    for grid in iter_grids():
        c = grid.point_to_cell(grid.get_random_point())
        c2 = grid.point_to_cell(grid.cell_to_point(c))
        np.testing.assert_almost_equal(c, c2)
        
        p_emtpy = np.zeros((0, grid.num_axes))
        assert grid.point_to_cell(p_emtpy).size == 0
        assert grid.cell_to_point(p_emtpy).size == 0
        
        
        
def test_grid_volume():
    """ test calculation of the grid volume """
    for grid in iter_grids():
        vol_discr = np.broadcast_to(grid.cell_volume_data, grid.shape).sum()
        assert vol_discr == pytest.approx(grid.volume)



@skipUnlessModule('matplotlib')
def test_grid_plotting():
    """ test plotting of grids """
    grids.UnitGrid([4]).plot()
    grids.UnitGrid([4, 4]).plot()
    
    with pytest.raises(NotImplementedError):
        grids.UnitGrid([4, 4, 4]).plot()

    grids.PolarGrid(4, 8).plot()
    grids.PolarGrid((2, 4), 8).plot()
