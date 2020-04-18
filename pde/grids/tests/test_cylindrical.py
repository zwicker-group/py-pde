'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np
import pytest

from .. import CylindricalGrid



def test_cylindrical_grid():
    """ test simple cylindrical grid """
    for periodic in [True, False]:
        grid = CylindricalGrid(4, (-1, 2), (8, 9), periodic_z=periodic)
        
        msg = str(grid)
        assert grid.dim == 3
        assert grid.numba_type == "f8[:, :]"
        assert grid.shape == (8, 9)
        assert grid.length == pytest.approx(3)
        assert grid.discretization[0] == pytest.approx(0.5)
        assert grid.discretization[1] == pytest.approx(1/3)
        np.testing.assert_array_equal(grid.discretization,
                                      np.array([0.5, 1/3]))
        assert grid.volume == pytest.approx(np.pi * 4**2 * 3)
        assert grid.volume == pytest.approx(grid.integrate(1))
        
        rs, zs = grid.axes_coords
        np.testing.assert_allclose(rs, np.linspace(0.25, 3.75, 8))
        np.testing.assert_allclose(zs, np.linspace(-1 + 1/6, 2 - 1/6, 9))
        
        # random points
        c = np.random.randint(8, size=(6, 2))
        c1 = grid.point_to_cell(grid.cell_to_point(c))
        np.testing.assert_almost_equal(c, c1, err_msg=msg)
        
        assert grid.contains_point(grid.get_random_point())
        assert grid.contains_point(grid.get_random_point(1.49))
        assert "laplace" in grid.operators
        


def test_cylindrical_to_cartesian():
    """ test conversion of cylindrical grid to Cartesian """
    from ...fields import ScalarField
    from .. import CartesianGrid
     
    expr_cyl = 'cos(z / 2) / (1 + r**2)'
    expr_cart = expr_cyl.replace('r**2', '(x**2 + y**2)')
     
    z_range = (-np.pi, 2*np.pi)
    grid_cyl = CylindricalGrid(10, z_range, (16, 33))
    pf_cyl = ScalarField.from_expression(grid_cyl, expression=expr_cyl)
    
    grid_cart = CartesianGrid([[-7, 7], [-6, 7], z_range], [16, 16, 16])
    pf_cart1 = pf_cyl.interpolate_to_grid(grid_cart)
    pf_cart2 = ScalarField.from_expression(grid_cart, expression=expr_cart) 
    np.testing.assert_allclose(pf_cart1.data, pf_cart2.data, atol=0.1)



def test_setting_domain_cylindrical():
    """ test various versions of settings bcs for cylindrical grids """
    grid = CylindricalGrid(1, [0, 1], [2, 2], periodic_z=False)
    grid.get_boundary_conditions('natural')
    grid.get_boundary_conditions(['no-flux', 'no-flux'])
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(['no-flux'])
    with pytest.raises(ValueError):
        grid.get_boundary_conditions(['no-flux'] * 3)
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions(['no-flux', 'periodic'])

    grid = CylindricalGrid(1, [0, 1], [2, 2], periodic_z=True)
    grid.get_boundary_conditions('natural')
    grid.get_boundary_conditions(['no-flux', 'periodic'])
    with pytest.raises(RuntimeError):
        grid.get_boundary_conditions(['no-flux', 'no-flux'])
        
        

@pytest.mark.parametrize('periodic', [True, False])
def test_polar_conversion(periodic):
    """ test conversion to polar coordinates """
    grid = CylindricalGrid(1, [-1, 1], [5, 5], periodic_z=periodic)
    dists = grid.polar_coordinates_real([0, 0, 0])
    assert np.all(0.09 <= dists) 
    assert np.any(dists < 0.11)
    assert np.all(dists <= np.sqrt(2)) 
    assert np.any(dists > 0.8 * np.sqrt(2)) 
        
