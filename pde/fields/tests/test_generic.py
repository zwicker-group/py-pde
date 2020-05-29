'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np
from scipy import ndimage
import pytest

from .. import ScalarField, VectorField, Tensor2Field, FieldCollection
from ..base import FieldBase
from ...grids import (UnitGrid, CartesianGrid, PolarGrid, SphericalGrid,
                      CylindricalGrid)
from ...grids.cartesian import CartesianGridBase
from ...tools.misc import skipUnlessModule



def iter_grids():
    """ generator providing some test grids """
    for periodic in [True, False]:
        yield UnitGrid([3], periodic=periodic)
        yield UnitGrid([3, 3, 3], periodic=periodic)
        yield CartesianGrid([[-1, 2], [0, 3]], [5, 7], periodic=periodic)
        yield CylindricalGrid(3, [-1, 2], [7, 8], periodic_z=periodic)
    yield PolarGrid(3, 4)
    yield SphericalGrid(3, 4)
    
    

@pytest.mark.parametrize("grid", iter_grids())
@pytest.mark.parametrize("field_class", [ScalarField, Tensor2Field])
def test_interpolation_natural(grid, field_class):
    """ test some interpolation for natural boundary conditions """
    msg = f'grid={grid}, field={field_class}'
    f = field_class.random_uniform(grid)
    if isinstance(grid, CartesianGridBase):
        p = grid.get_random_point(boundary_distance=.5)
    else:
        p = grid.get_random_point(boundary_distance=1, avoid_center=True)
    p = grid.point_from_cartesian(p)
    i1 = f.interpolate(p, method='scipy_linear')
    i2 = f.interpolate(p, method='numba')
    np.testing.assert_almost_equal(i1, i2, err_msg=msg)

    c = (1,) * len(grid.axes)  # specific cell
    p = f.grid.cell_coords[c]
    np.testing.assert_allclose(f.interpolate(p, method='scipy_linear'),
                               f.data[(Ellipsis,) + c], err_msg=msg)
    np.testing.assert_allclose(f.interpolate(p, method='numba'),
                               f.data[(Ellipsis,) + c], err_msg=msg)


           
@pytest.mark.parametrize("num", [1, 3])
@pytest.mark.parametrize("grid", iter_grids())
def test_shapes_nfields(num, grid):
    """ test single component field """
    fields = [ScalarField.random_uniform(grid)
              for _ in range(num)]
    field = FieldCollection(fields)
    data_shape = (num, ) + grid.shape
    np.testing.assert_equal(field.data.shape, data_shape)
    for pf_single in field:
        np.testing.assert_equal(pf_single.data.shape, grid.shape)
     
    field_c = field.copy()
    np.testing.assert_allclose(field.data, field_c.data)
    assert field.grid == field_c.grid



def test_arithmetics():
    """ test simple arithmetics for fields """
    grid = UnitGrid([2, 2])
    for cls in (ScalarField, VectorField, Tensor2Field):
        f1 = cls(grid, data=1)
        f2 = cls(grid, data=2)
        assert isinstance(str(f1), str)
        np.testing.assert_allclose(f1.data, 1)
        
        np.testing.assert_allclose((-f1).data, -1)

        # test addition
        np.testing.assert_allclose((f1 + 1).data, 2)
        np.testing.assert_allclose((1 + f1).data, 2)
        f1 += 1
        np.testing.assert_allclose(f1.data, 2)
        np.testing.assert_allclose((f1 + f2).data, 4)
        
        # test subtraction
        np.testing.assert_allclose((f1 - 1).data, 1)
        np.testing.assert_allclose((1 - f1).data, -1)
        f1 -= 1
        np.testing.assert_allclose(f1.data, 1)
        np.testing.assert_allclose((f1 - f2).data, -1)

        # test multiplication
        np.testing.assert_allclose((f1 * 2).data, 2)
        np.testing.assert_allclose((2 * f1).data, 2)
        f1 *= 2 
        np.testing.assert_allclose(f1.data, 2)
        
        # test division
        np.testing.assert_allclose((f1 / 2).data, 1)
        with pytest.raises(TypeError):
            np.testing.assert_allclose((2 / f1).data, 1)
        f1 /= 2
        np.testing.assert_allclose(f1.data, 1)

        # test power
        f1.data = 2
        np.testing.assert_allclose((f1**3).data, 8)
        f1 **= 3
        np.testing.assert_allclose(f1.data, 8)
        
        # test applying a function
        f1.data = 2
        np.testing.assert_allclose(f1.apply(lambda x: x**3).data, 8)
        f1.apply(lambda x: x**3, out=f1)
        np.testing.assert_allclose(f1.data, 8)



def test_scalar_arithmetics():
    """ test simple arithmetics involving scalar fields """
    grid = UnitGrid([3, 4])
    s = ScalarField(grid, data=2)
    v = VectorField.random_uniform(grid)
    
    for f in [v, FieldCollection([v])]:
        f.data = s
        assert f.data.shape == (2, 3, 4)
        np.testing.assert_allclose(f.data, 2)
        
        f += s
        np.testing.assert_allclose(f.data, 4)
        np.testing.assert_allclose((f + s).data, 6)
        np.testing.assert_allclose((s + f).data, 6)
        f -= s
        np.testing.assert_allclose((f - s).data, 0)
        np.testing.assert_allclose((s - f).data, 0)
        
        f *= s
        np.testing.assert_allclose(f.data, 4)
        np.testing.assert_allclose((f * s).data, 8)
        np.testing.assert_allclose((s * f).data, 8)
        f /= s
        np.testing.assert_allclose((f / s).data, 1)
        with pytest.raises(TypeError):
            s / f
        with pytest.raises(TypeError):
            s /= f
        with pytest.raises(TypeError):
            s *= f
        


def test_data_managment():
    """ test how data is set """
    grid = UnitGrid([2, 2])
    for cls in (ScalarField, VectorField, Tensor2Field):
        s1 = cls(grid, data=1)
        np.testing.assert_allclose(s1.data, 1)
        
        s2 = cls(grid)
        np.testing.assert_allclose(s2.data, 0)
        
        c = FieldCollection([s1, s2])
        s1.data = 0
        np.testing.assert_allclose(c.data, 0)
        
        c.data = 2
        np.testing.assert_allclose(s1.data, 2)
        np.testing.assert_allclose(s2.data, 2)
        
        c.data += 1
        np.testing.assert_allclose(s1.data, 3)
        np.testing.assert_allclose(s2.data, 3)
        
        c[0].data += 2  # reference to s1
        c[1].data *= 2  # reference to s2
        np.testing.assert_allclose(s1.data, 5)
        np.testing.assert_allclose(s2.data, 6)
        
        c[0] = s2
        np.testing.assert_allclose(c.data, 6)
        
        # nested collections
        with pytest.raises(RuntimeError):
            FieldCollection([c])


            
@skipUnlessModule("h5py")
def test_hdf_input_output(tmp_path):
    """ test writing and reading files """
    grid = UnitGrid([4, 4])
    s = ScalarField.random_uniform(grid, label='scalar')
    v = VectorField.random_uniform(grid, label='vector')
    t = Tensor2Field.random_uniform(grid, label='tensor')
    col = FieldCollection([s, v, t], label='collection')
    
    path = tmp_path / 'test_hdf_input_output.hdf5'
    for f in [s, v, t, col]:
        f.to_file(path)
        f2 = FieldBase.from_file(path)
        assert f == f2
        assert f.label == f2.label
        assert isinstance(str(f), str)
        assert isinstance(repr(f), str)
            
      
      
@skipUnlessModule("matplotlib")
def test_writing_images(tmp_path):
    """ test writing and reading files """
    from matplotlib.pyplot import imread
    
    grid = UnitGrid([4, 4])
    s = ScalarField.random_uniform(grid, label='scalar')
    v = VectorField.random_uniform(grid, label='vector')
    t = Tensor2Field.random_uniform(grid, label='tensor')
    
    path = tmp_path / 'test_writing_images.png'
    for f in [s, v, t]:
        f.to_file(path)
        # try reading the file
        with path.open('br') as fp:
            imread(fp)
      
           
           
def test_interpolation_to_grid_fields():
    """ test whether data is interpolated correctly for different fields """
    grid = CartesianGrid([[0, 2*np.pi]]*2, 6)
    grid2 = CartesianGrid([[0, 2*np.pi]]*2, 8)
    vf = VectorField.from_expression(grid, ['sin(y)', 'cos(x)'])
    sf = vf[0]  # test extraction of fields
    fc = FieldCollection([sf, vf])
    
    for f in [sf, vf, fc]:
        f2 = f.interpolate_to_grid(grid2, method='numba')
        f3 = f2.interpolate_to_grid(grid, method='numba')
        np.testing.assert_allclose(f.data, f3.data, atol=0.2, rtol=0.2)
            
      
           
@pytest.mark.parametrize('field_cls', [ScalarField, VectorField, Tensor2Field])
def test_interpolation_values(field_cls):
    """ test whether data is interpolated correctly for different fields """
    grid = UnitGrid([3, 4])
    f = field_cls.random_uniform(grid)
    
    intp = f.make_interpolator('numba')
    c = f.grid.cell_coords[2, 2]
    np.testing.assert_allclose(intp(c), f.data[..., 2, 2])
    
    with pytest.raises(ValueError):
        intp(np.array([100, -100]))

    res = f.make_interpolator('numba', fill=45)(np.array([100, -100]))
    np.testing.assert_almost_equal(res, np.full(f.data_shape, 45))



@pytest.mark.parametrize('grid', [UnitGrid([6]),
                                  PolarGrid(6, 4),
                                  SphericalGrid(7, 4),
                                  CylindricalGrid(6, (0, 8), (7, 8))])
def test_interpolation_to_cartesian(grid):
    """ test whether data is interpolated correctly to Cartesian grid """
    dim = grid.dim
    vf = VectorField(grid, 2)
    sf = vf[0]  # test extraction of fields
    fc = FieldCollection([sf, vf])
    
    # subset
    grid_cart = UnitGrid([4] * dim)
    for f in [sf, fc]:
        res = f.interpolate_to_grid(grid_cart)
        np.testing.assert_allclose(res.data, 2)
    
    # superset
    grid_cart = UnitGrid([8] * dim)
    for f in [sf, fc]:
        res = f.interpolate_to_grid(grid_cart, fill=0)
        assert res.data.min() == 0
        assert res.data.max() == pytest.approx(2)
        
        
        
@pytest.mark.parametrize('grid', [PolarGrid(6, 4),
                                  SphericalGrid(7, 4),
                                  CylindricalGrid(6, (0, 8), (7, 8))])
def test_get_cartesian_grid(grid):
    """ test whether Cartesian grids can be created """
    cart = grid.get_cartesian_grid(mode='valid')
    assert cart.volume < grid.volume
    cart = grid.get_cartesian_grid(mode='full')
    assert cart.volume > grid.volume
        

        
@skipUnlessModule("matplotlib")
@pytest.mark.parametrize("grid", iter_grids())
def test_simple_plotting(grid):
    """ test simple plotting of various fields on various grids """
    import matplotlib.pyplot as plt
    
    vf = VectorField.random_uniform(grid)
    tf = Tensor2Field.random_uniform(grid)
    sf = tf[0, 0]  # test extraction of fields
    fc = FieldCollection([sf, vf])
    for f in [sf, vf, tf, fc]:
        f.plot()
        f.plot('line')
        if grid.dim >= 2:
            f.plot('image')
        if isinstance(f, VectorField) and grid.dim == 2:
            f.plot('quiver')
            f.plot('streamplot')
        plt.close('all')
        
            
            
def test_random_uniform():
    """ test whether random uniform fields behave correctly """
    grid = UnitGrid([256, 256])
    for field_cls in [ScalarField, VectorField, Tensor2Field]:
        a = np.random.random()
        b = 2 + np.random.random()
        f = field_cls.random_uniform(grid, a, b)
        assert np.mean(f.average) == pytest.approx((a + b) / 2, rel=0.02)
        assert np.std(f.data) == pytest.approx(0.288675 * (b - a), rel=0.1)
    
    
    
def test_random_normal():
    """ test whether random normal fields behave correctly """
    grid = UnitGrid([256, 256])
    for field_cls in [ScalarField, VectorField, Tensor2Field]:
        m = np.random.random()
        s = 1 + np.random.random()
        for scaling in ['none', 'physical']:
            f = field_cls.random_normal(grid, mean=m, std=s,
                                        scaling=scaling)
            assert np.mean(f.average) == pytest.approx(m, rel=0.1, abs=0.1)
            assert np.std(f.data) == pytest.approx(s, rel=0.1, abs=0.1)
            
    
    
@pytest.mark.parametrize('field_cls', [ScalarField, VectorField, Tensor2Field])
def test_random_colored(field_cls):
    """ test whether random colored fields behave correctly """
    grid = UnitGrid([128, 128])
    exponent = np.random.uniform(-4, 4)
    scale = 1 + np.random.random()
    f = field_cls.random_colored(grid, exponent=exponent, scale=scale)
    
    assert np.allclose(f.average, 0)
            
            

def test_fluctuations():
    """ test the scaling of fluctuations """
    for dim in [1, 2]:
        for size in [256, 512]:
            if dim == 1:
                size **= 2
            grid = CartesianGrid([[0, 1]] * dim, [size] * dim)
            std = 1 + np.random.random()
            for field_cls in [ScalarField, VectorField, Tensor2Field]:
                s = field_cls.random_normal(grid, mean=np.random.random(),
                                            std=std, scaling='physical')
                expect = np.full([dim] * field_cls.rank, std)
                np.testing.assert_allclose(s.fluctuations, expect, rtol=0.1)
             
             

def test_smoothing():
    """ test smoothing on different grids """
    for grid in [CartesianGrid([[-2, 3]], 4),
                 UnitGrid(7, periodic=False), UnitGrid(7, periodic=True)]:
        f1 = ScalarField.random_uniform(grid)
        sigma = 0.5 + np.random.random()
        
        # this assumes that the grid periodicity is the same for all axes
        mode = 'wrap' if grid.periodic[0] else 'reflect'         
        s = sigma / grid.typical_discretization   
        expected = ndimage.gaussian_filter(f1.data, sigma=s, mode=mode)
        
        out = f1.smooth(sigma) 
        np.testing.assert_allclose(out.data, expected)
        
        out.data = 0  # reset data
        f1.smooth(sigma, out=out).data 
        np.testing.assert_allclose(out.data, expected)

    # test one simple higher order smoothing
    tf = Tensor2Field.random_uniform(grid)
    assert tf.data.shape == tf.smooth(1).data.shape
    
    # test in-place smoothing
    g = UnitGrid([8, 8])
    f1 = ScalarField.random_normal(g)
    f2 = f1.smooth(3)
    f1.smooth(3, out=f1)
    np.testing.assert_allclose(f1.data, f2.data)



def test_vector_from_scalars():
    """ test how to compile vector fields from scalar fields """
    g = UnitGrid([1, 2])
    s1 = ScalarField(g, [[0, 1]])
    s2 = ScalarField(g, [[2, 3]])
    v = VectorField.from_scalars([s1, s2], label='test')
    assert v.label == 'test'
    np.testing.assert_equal(v.data, [[[0, 1]], [[2, 3]]])
    
    with pytest.raises(ValueError):
        VectorField.from_scalars([s1, s2, s1])



def test_dot_product():
    """ test dot products between vectors and tensors """
    g = UnitGrid([3, 2])
    vf = VectorField.random_normal(g)
    tf = Tensor2Field.random_normal(g)
    v_dot = vf.get_dot_operator()
    t_dot = tf.get_dot_operator()

    expected = np.einsum('i...,i...->...', vf.data, vf.data)
    np.testing.assert_allclose((vf @ vf).data, expected)
    np.testing.assert_allclose(v_dot(vf.data, vf.data), expected)
    
    expected = np.einsum('i...,i...->...', vf.data, tf.data)
    np.testing.assert_allclose((vf @ tf).data, expected)
    np.testing.assert_allclose(v_dot(vf.data, tf.data), expected)

    expected = np.einsum('ji...,i...->j...', tf.data, vf.data)
    np.testing.assert_allclose((tf @ vf).data, expected)
    np.testing.assert_allclose(t_dot(tf.data, vf.data), expected)

    expected = np.einsum('ij...,jk...->ik...', tf.data, tf.data)
    np.testing.assert_allclose((tf @ tf).data, expected)
    np.testing.assert_allclose(t_dot(tf.data, tf.data), expected)
    
