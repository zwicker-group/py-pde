'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import tempfile

import numpy as np
import pytest

from .test_generic import iter_grids
from ..scalar import ScalarField
from ..base import FieldBase
from ...grids import UnitGrid, CartesianGrid, PolarGrid
from ...grids.tests.test_cartesian import _get_cartesian_grid
from ...tools.misc import module_available, skipUnlessModule



def test_interpolation_singular():
    """ test interpolation on singular dimensions """
    grid = UnitGrid([1])
    field = ScalarField(grid, data=3)
    
    # test constant boundary conditions
    bc = [{'type': 'value', 'value': 1}, {'type': 'value', 'value': 5}]
    x = np.linspace(0, 1, 7).reshape((7, 1))
    y = field.interpolate(x, method='numba_linear', bc=bc)
    np.testing.assert_allclose(y, 1 + 4*x.ravel())

    # test derivative boundary conditions
    bc = [{'type': 'derivative', 'value': -2},
          {'type': 'derivative', 'value': 2}]
    x = np.linspace(0, 1, 7).reshape((7, 1))
    y = field.interpolate(x, method='numba_linear', bc=bc)
    np.testing.assert_allclose(y, 2 + 2*x.ravel())



@pytest.mark.parametrize("grid", iter_grids())
def test_simple_shapes(grid):
    """ test simple scalar fields """
    pf = ScalarField.random_uniform(grid)
    np.testing.assert_equal(pf.data.shape, grid.shape)
    pf_lap = pf.laplace('natural')
    np.testing.assert_equal(pf_lap.data.shape, grid.shape)
    assert isinstance(pf.integral, float)
    
    pf_c = pf.copy()
    np.testing.assert_allclose(pf.data, pf_c.data)
    assert pf.grid == pf_c.grid
    assert pf.data is not pf_c.data
                
    if module_available("matplotlib"):
        pf.plot()  # simply test whether this does not cause errors


           
def test_scalars():
    """ test some scalar fields """
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])
    s1 = ScalarField(grid, np.full(grid.shape, 1))
    s2 = ScalarField(grid, np.full(grid.shape, 2))
    assert s1.average == pytest.approx(1)
    assert s1.magnitude == pytest.approx(1)
    
    s3 = s1 + s2
    assert s3.grid == grid
    np.testing.assert_allclose(s3.data, 3)
    s1 += s2
    np.testing.assert_allclose(s1.data, 3)
    
    s2 = FieldBase.from_state(s1.state_serialized, grid=grid, data=s1.data)
    assert s1 == s2



def test_laplacian():
    """ test the gradient operator """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16],
                         periodic=True)
    s = ScalarField.random_harmonic(grid, axis_combination=np.add, modes=1)
    s_lap = s.laplace('natural')
    assert s_lap.data.shape == (16, 16)
    np.testing.assert_allclose(s_lap.data, -s.data, rtol=0.1, atol=0.1)



def test_gradient():
    """ test the gradient operator """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16],
                         periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = np.cos(x) + np.sin(y)
    s = ScalarField(grid, data)
    v = s.gradient('natural')
    assert v.data.shape == (2, 16, 16)
    np.testing.assert_allclose(v.data[0], -np.sin(x), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(v.data[1], np.cos(y), rtol=0.1, atol=0.1)
   
        
        
@pytest.mark.parametrize("grid", iter_grids())
def test_interpolation_to_grid(grid):
    """ test whether data is interpolated correctly for different grids """
    sf = ScalarField.random_uniform(grid)
    for normalized in [True, False]:
        sf2 = sf.interpolate_to_grid(grid, normalized=normalized,
                                     method='nearest')
        np.testing.assert_allclose(sf.data, sf2.data, rtol=1e-6,
                                   err_msg='grid=%s' % grid)


        
@pytest.mark.parametrize("grid", iter_grids())
def test_add_interpolated_scalar(grid):
    """ test the `add_interpolated` method """
    f = ScalarField(grid)
    a = np.random.random()
    
    c = tuple(grid.point_to_cell(grid.get_random_point()))
    p = grid.cell_to_point(c, cartesian=False)
    f.add_interpolated(p, a)
    assert f.data[c] == pytest.approx(a / grid.cell_volumes[c])
    
    f.add_interpolated(grid.get_random_point(cartesian=False), a)
    assert f.integral == pytest.approx(2 * a)
    
    f.data = 0  # reset
    add_interpolated = grid.make_add_interpolated_compiled()
    c = tuple(grid.point_to_cell(grid.get_random_point()))
    p = grid.cell_to_point(c, cartesian=False)
    add_interpolated(f.data, p, a)
    assert f.data[c] == pytest.approx(a / grid.cell_volumes[c])

    add_interpolated(f.data, grid.get_random_point(cartesian=False), a)
    assert f.integral == pytest.approx(2 * a)



def test_add_interpolated_1d():
    """ test the `add_interpolated` method for 1d systems """
    grid = PolarGrid(3, 5)
    f = ScalarField(grid)
    g = f.copy()
    a = np.random.random()
    for r in np.linspace(0, 3, 8).reshape(8, 1):
        f.data = g.data = 0
        f.add_interpolated(r, a)
        assert f.integral == pytest.approx(a)
        grid.make_add_interpolated_compiled()(g.data, r, a)
        np.testing.assert_array_almost_equal(f.data, g.data)

        
        
def test_random_harmonic():
    """ test whether random harmonic fields behave correctly """
    grid = _get_cartesian_grid(2)  # get random Cartesian grid
    x = ScalarField.random_harmonic(grid, modes=1)
    scaling = sum((2 * np.pi / L)**2 for L in grid.cuboid.size)
    y = -x.laplace('natural') / scaling
    np.testing.assert_allclose(x.data, y.data, rtol=1e-2, atol=1e-2)
    


def test_get_line_data():
    """ test different extraction methods for line data """
    grid = UnitGrid([16, 32])
    c = ScalarField.random_harmonic(grid)
    
    np.testing.assert_equal(c.get_line_data(extract='cut_0'),
                            c.get_line_data(extract='cut_x'))
    np.testing.assert_equal(c.get_line_data(extract='cut_1'),
                            c.get_line_data(extract='cut_y'))
    
    for extract in ['project_0', 'project_1', 'project_x', 'project_y']:
        data = c.get_line_data(extract=extract)['data_y']
        np.testing.assert_allclose(data, 0, atol=1e-14)
        


def test_from_expression():
    """ test creating scalar field from expression """
    grid = UnitGrid([1, 2])
    sf = ScalarField.from_expression(grid, "x * y")
    np.testing.assert_allclose(sf.data, [[0.25, 0.75]])
    
    
    
def test_interpolation_inhomogeneous_bc():
    """ test field interpolation with inhomogeneous boundary condition """
    sf = ScalarField(UnitGrid([3, 3], periodic=False))
    x = 1 + np.random.random()
    v = sf.interpolate([x, 0], method='numba_linear',
                       bc=['natural', {'type': 'value', 'value': 'x'}])
    assert x == pytest.approx(v)



@skipUnlessModule("matplotlib")
def test_from_image():
    from matplotlib.pyplot import imsave
    img_data = np.random.uniform(size=(9, 8, 3))
    img_data_gray = img_data @ np.array([0.299, 0.587, 0.114])
    with tempfile.NamedTemporaryFile(suffix='.png') as fp:
        imsave(fp.name, img_data, vmin=0, vmax=1)
        sf = ScalarField.from_image(fp.name)
    np.testing.assert_allclose(sf.data, img_data_gray.T[:, ::-1], atol=0.05)
    


def test_to_scalar():
    """ test conversion to scalar field """
    sf = ScalarField.random_uniform(UnitGrid([3, 3]))
    np.testing.assert_allclose(sf.to_scalar().data, sf.data)    
    np.testing.assert_allclose(sf.to_scalar('squared_sum').data, sf.data**2)
