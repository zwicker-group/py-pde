'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import random

import numpy as np
import pytest

from .. import UnitGrid, CartesianGrid
from ..boundaries import Boundaries, PeriodicityError, DomainError



def _get_cartesian_grid(dim=2, periodic=True):
    """ return a random Cartesian grid of given dimension """
    bounds = [[0, 1 + np.random.random()] for _ in range(dim)]
    shape = np.random.randint(32, 64, size=dim)
    return CartesianGrid(bounds, shape, periodic=periodic)



def test_degenerated_grid():
    """ test degenerated grids """
    with pytest.raises(ValueError):
        UnitGrid([])
    with pytest.raises(ValueError):
        CartesianGrid([], 1)



def test_generic_cartesian_grid():
    """ test generic cartesian grid functions """
    for dim in (1, 2, 3):
        periodic = random.choices([True, False], k=dim)
        shape = np.random.randint(2, 8, size=dim)
        a = np.random.random(dim)
        b = a + np.random.random(dim)
        
        cases = [UnitGrid(shape, periodic=periodic),
                 CartesianGrid(np.c_[a, b], shape, periodic=periodic)]
        for grid in cases:
            assert grid.dim == dim
            dim_axes = len(grid.axes) + len(grid.axes_symmetric)
            assert dim_axes == dim
            vol = np.prod(grid.discretization) * np.prod(shape)
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
            assert "laplace" in grid.operators



@pytest.mark.parametrize("periodic", [True, False])
def test_unit_grid_1d(periodic):
    """ test 1D grids """
    grid = UnitGrid(4, periodic=periodic)
    assert grid.dim == 1
    assert grid.numba_type == "f8[:]"
    assert grid.volume == 4
    np.testing.assert_array_equal(grid.discretization, np.ones(1))
    dist, angle = grid.polar_coordinates_real(0, ret_angle=True)
    if periodic:
        np.testing.assert_allclose(dist, [0.5, 1.5, 1.5, 0.5])
    else:
        np.testing.assert_allclose(dist, np.arange(4) + 0.5)
    assert angle.shape == (4,)
    
    grid = UnitGrid(8, periodic=periodic)
    assert grid.dim == 1
    assert grid.volume == 8

    norm_numba = grid.make_normalize_point_compiled()
    
    def norm_numba_wrap(x):
        y = np.array([x])
        norm_numba(y)
        return y
    
    for normalize in [grid.normalize_point, norm_numba_wrap]:
        if periodic:
            np.testing.assert_allclose(normalize(-1e-10), 8 - 1e-10)
            np.testing.assert_allclose(normalize(1e-10), 1e-10)
            np.testing.assert_allclose(normalize(8 - 1e-10), 8 - 1e-10)
            np.testing.assert_allclose(normalize(8 + 1e-10), 1e-10)
        else:
            for x in [-1e-10, 1e-10, 8 - 1e-10, 8 + 1e-10]:
                np.testing.assert_allclose(normalize(x), x)

    grid = UnitGrid(8, periodic=periodic)
    
    # test conversion between polar and Cartesian coordinates
    c1 = grid.cell_coords
    p = np.random.random(1) * grid.shape
    d, a = grid.polar_coordinates_real(p, ret_angle=True)
    c2 = grid.from_polar_coordinates(d, a, p)
    assert np.allclose(grid.distance_real(c1, c2), 0)
    
    # test boundary points
    np.testing.assert_equal(grid._boundary_coordinates(0, False), np.array([0]))
    np.testing.assert_equal(grid._boundary_coordinates(0, True), np.array([8]))
    
    
    
def test_unit_grid_2d():
    """ test 2D grids """
    # test special case
    grid = UnitGrid([4, 4], periodic=True)
    assert grid.dim == 2
    assert grid.numba_type == "f8[:, :]"
    assert grid.volume == 16
    np.testing.assert_array_equal(grid.discretization, np.ones(2))
    assert grid.get_image_data(np.zeros(grid.shape))['extent'] == [0, 4, 0, 4]
    for _ in range(10):
        p = np.random.randn(2)
        assert np.all(grid.polar_coordinates_real(p) < np.sqrt(8))
    large_enough = grid.polar_coordinates_real((0, 0)) > np.sqrt(4)
    assert np.any(large_enough)
    
    periodic = random.choices([True, False], k=2)
    grid = UnitGrid([4, 4], periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 16
    assert grid.polar_coordinates_real((1, 1)).shape == (4, 4)
    
    grid = UnitGrid([4, 8], periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 32
    assert grid.polar_coordinates_real((1, 1)).shape == (4, 8)
    
    # test conversion between polar and Cartesian coordinates
    c1 = grid.cell_coords
    p = np.random.random(2) * grid.shape
    d, a = grid.polar_coordinates_real(p, ret_angle=True)
    c2 = grid.from_polar_coordinates(d, a, p)
    assert np.allclose(grid.distance_real(c1, c2), 0)
    
    # test boundary points
    np.testing.assert_equal(grid._boundary_coordinates(0, False),
                            np.c_[np.full(8, 0), np.linspace(0.5, 7.5, 8)])
    np.testing.assert_equal(grid._boundary_coordinates(0, True),
                            np.c_[np.full(8, 4), np.linspace(0.5, 7.5, 8)])
    np.testing.assert_equal(grid._boundary_coordinates(1, False),
                            np.c_[np.linspace(0.5, 3.5, 4), np.full(4, 0)])
    np.testing.assert_equal(grid._boundary_coordinates(1, True),
                            np.c_[np.linspace(0.5, 3.5, 4), np.full(4, 8)])



def test_unit_grid_3d():
    """ test 3D grids """
    grid = UnitGrid([4, 4, 4])
    assert grid.dim == 3
    assert grid.numba_type == "f8[:, :, :]"
    assert grid.volume == 64
    np.testing.assert_array_equal(grid.discretization, np.ones(3))
    assert grid.get_image_data(np.zeros(grid.shape))['extent'] == [0, 4, 0, 4]
    assert grid.polar_coordinates_real((1, 1, 3)).shape == (4, 4, 4)

    periodic = random.choices([True, False], k=3)
    grid = UnitGrid([4, 6, 8], periodic=periodic)
    assert grid.dim == 3
    assert grid.volume == 192
    assert grid.polar_coordinates_real((1, 1, 2)).shape == (4, 6, 8)

    grid = UnitGrid([4, 4, 4], periodic=True)
    assert grid.dim == 3
    assert grid.volume == 64
    for _ in range(10):
        p = np.random.randn(3)
        not_too_large = grid.polar_coordinates_real(p) < np.sqrt(12)
        assert np.all(not_too_large)
    large_enough = grid.polar_coordinates_real((0, 0, 0)) > np.sqrt(6)
    assert np.any(large_enough)

    # test boundary points
    for bndry in grid._iter_boundaries():
        assert grid._boundary_coordinates(*bndry).shape == (4, 4, 3)



def test_rect_grid_1d():
    """ test 1D grids """
    grid = CartesianGrid([32], 16, periodic=False)
    assert grid.dim == 1
    assert grid.volume == 32
    assert grid.typical_discretization == 2
    np.testing.assert_array_equal(grid.discretization, np.full(1, 2))
    assert grid.polar_coordinates_real(0).shape == (16,)
    
    grid = CartesianGrid([[-16, 16]], 8, periodic=True)
    assert grid.cuboid.pos == [-16]
    assert grid.shape == (8,)
    assert grid.dim == 1
    assert grid.volume == 32
    assert grid.typical_discretization == 4
    assert grid.polar_coordinates_real(1).shape == (8,)

    np.testing.assert_allclose(grid.normalize_point(-16 - 1e-10),
                               16 - 1e-10)
    np.testing.assert_allclose(grid.normalize_point(-16 + 1e-10),
                               -16 + 1e-10)
    np.testing.assert_allclose(grid.normalize_point(16 - 1e-10),
                               16 - 1e-10)
    np.testing.assert_allclose(grid.normalize_point(16 + 1e-10),
                               -16 + 1e-10)
            
    for periodic in [True, False]:
        a, b = np.random.random(2)
        grid = CartesianGrid([[a, a + b]], 8, periodic=periodic)
        
        # test conversion between polar and Cartesian coordinates
        c1 = grid.cell_coords
        p = np.random.random(1) * grid.shape
        d, a = grid.polar_coordinates_real(p, ret_angle=True)
        c2 = grid.from_polar_coordinates(d, a, p)
        assert np.allclose(grid.distance_real(c1, c2), 0)


    
def test_rect_grid_2d():
    """ test 2D grids """
    grid = CartesianGrid([[2], [2]], 4, periodic=True)
    assert grid.get_image_data(np.zeros(grid.shape))['extent'] == [0, 2, 0, 2]
    for _ in range(10):
        p = np.random.randn(2)
        assert np.all(grid.polar_coordinates_real(p) < np.sqrt(2))
        
    periodic = random.choices([True, False], k=2)
    grid = CartesianGrid([[4], [4]], 4, periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 16
    np.testing.assert_array_equal(grid.discretization, np.ones(2))
    assert grid.typical_discretization == 1
    assert grid.polar_coordinates_real((1, 1)).shape == (4, 4)
    
    grid = CartesianGrid([[-2, 2], [-2, 2]], [4, 8],
                               periodic=periodic)
    assert grid.dim == 2
    assert grid.volume == 16
    assert grid.typical_discretization == 0.75
    assert grid.polar_coordinates_real((1, 1)).shape == (4, 8)
    
    # test conversion between polar and Cartesian coordinates
    c1 = grid.cell_coords
    p = np.random.random(2) * grid.shape
    d, a = grid.polar_coordinates_real(p, ret_angle=True)
    c2 = grid.from_polar_coordinates(d, a, p)
    
    assert np.allclose(grid.distance_real(c1, c2), 0)



def test_rect_grid_3d():
    """ test 3D grids """
    grid = CartesianGrid([4, 4, 4], 4)
    assert grid.dim == 3
    assert grid.volume == 64
    assert grid.typical_discretization == 1
    np.testing.assert_array_equal(grid.discretization, np.ones(3))
    assert grid.polar_coordinates_real((1, 1, 3)).shape == (4, 4, 4)

    bounds = [[-2, 2], [-2, 2], [-2, 2]]
    grid = CartesianGrid(bounds, [4, 6, 8])
    assert grid.dim == 3
    np.testing.assert_allclose(grid.axes_bounds, bounds)
    assert grid.volume == 64
    assert grid.typical_discretization == pytest.approx(0.7222222222222)
    assert grid.polar_coordinates_real((1, 1, 2)).shape == (4, 6, 8)

    grid = CartesianGrid([[2], [2], [2]], 4, periodic=True)
    for _ in range(10):
        p = np.random.randn(3)
        assert np.all(grid.polar_coordinates_real(p) < np.sqrt(3))



def test_unit_rect_grid():
    """ test whether the rectangular grid behaves like a unit grid in
    special cases """
    for periodic in [True, False]:
        msg = 'periodic=%s' % str(periodic)
        
        dim = random.randrange(1, 4)
        shape = np.random.randint(2, 10, size=dim)
        g1 = UnitGrid(shape, periodic=periodic)
        g2 = CartesianGrid(np.c_[np.zeros(dim), shape], shape,
                                 periodic=periodic)
        volume = np.prod(shape)
        assert g1.volume == pytest.approx(volume)
        assert g2.volume == pytest.approx(volume)
        assert g1.integrate(1) == pytest.approx(volume)
        assert g2.integrate(1) == pytest.approx(volume)
        
        assert g1.dim == g2.dim == dim
        np.testing.assert_array_equal(g1.shape, g2.shape)
        assert (g1.typical_discretization ==
                pytest.approx(g2.typical_discretization))
        
        for _ in range(10):
            p1, p2 = np.random.normal(scale=10, size=(2, dim))
            assert (g1.distance_real(p1, p2) ==
                    pytest.approx(g2.distance_real(p1, p2)))
            
        p0 = np.random.normal(scale=10, size=dim)
        np.testing.assert_allclose(g1.polar_coordinates_real(p0),
                                   g2.polar_coordinates_real(p0),
                                   err_msg=msg)
        
        
        
def test_conversion_unit_rect_grid():
    """ test the conversion from unit to rectangular grid """
    dim = random.randrange(1, 4)
    shape = np.random.randint(2, 10, size=dim)
    periodic = random.choices([True, False], k=dim)
    g1 = UnitGrid(shape, periodic=periodic)
    g2 = g1.to_cartesian()
    
    assert g1.shape == g2.shape
    assert g1.cuboid == g2.cuboid
    assert g1.periodic == g2.periodic



def test_setting_boundary_conditions():
    """ test setting some boundary conditions """
    grid = UnitGrid([3, 3], periodic=[True, False])
    for bc in [grid.get_boundary_conditions('natural'),
               grid.get_boundary_conditions(['natural', 'no-flux'])]:
        assert isinstance(bc, Boundaries)
        
    for bc in ['periodic', 'value']:
        with pytest.raises(PeriodicityError):
            grid.get_boundary_conditions(bc)
        
        
            
def test_setting_domain_rect():
    """ test various versions of settings bcs for cartesian grids """
    grid = UnitGrid([2, 2])
    grid.get_boundary_conditions(['no-flux', 'no-flux'])

    # wrong number of conditions
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(['no-flux'])
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(['no-flux'] * 3)

    grid = UnitGrid([2, 2], periodic=[True, False])
    grid.get_boundary_conditions('natural')
    grid.get_boundary_conditions(['periodic', 'no-flux'])
        
    # incompatible conditions
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions('periodic')
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions('no-flux')
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions(['no-flux', 'periodic'])



def test_interpolate_1d():
    """ test interpolation of 1d grid """
    grid = UnitGrid(2, periodic=False)
    intp = grid.make_interpolator_compiled(bc={'type': 'value', 'value': 1})
    
    assert intp(np.zeros(2), np.zeros(1)) == pytest.approx(1)
    assert intp(np.zeros(2), np.ones(1)) == pytest.approx(0)
    with pytest.raises(DomainError):
        intp(np.zeros(2), np.array([-1]))
    with pytest.raises(DomainError):
        intp(np.zeros(2), np.array([3]))
    
    grid_per = UnitGrid(2, periodic=True)
    intp = grid_per.make_interpolator_compiled(bc='natural')
    for pos in [-1, 0, 1, 2, 3]:
        assert intp(np.arange(2), np.array([pos])) == pytest.approx(0.5)



def test_interpolate_2d():
    """ test interpolation of 2d grid """
    grid = UnitGrid([2, 2], periodic=False)
    intp = grid.make_interpolator_compiled(bc={'type': 'value', 'value': 1})
    
    assert intp(np.zeros((2, 2)), np.array([0, 1])) == pytest.approx(1)



def test_interpolate_3d():
    """ test interpolation of 3d grid """
    grid = UnitGrid([2, 2, 2], periodic=False)
    intp = grid.make_interpolator_compiled(bc={'type': 'value', 'value': 1})
    
    val = intp(np.zeros((2, 2, 2)), np.array([0, 1, 1]))
    assert val == pytest.approx(1)

