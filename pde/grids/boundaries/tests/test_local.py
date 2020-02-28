'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np
import pytest

from ..local import _get_arr_1d, BCBase
from ... import UnitGrid



def test_get_arr_1d():
    """ test the _get_arr_1d function """
    # 1d
    a = np.arange(3)
    arr_1d, i, bc_idx = _get_arr_1d(a, [1], 0)
    assert i == 1
    assert bc_idx == ()
    np.testing.assert_equal(arr_1d, a)

    # 2d
    a = np.arange(4).reshape(2, 2)
    arr_1d, i, bc_idx = _get_arr_1d(a, [0, 0], 0)
    assert i == 0
    assert bc_idx == (0,)
    np.testing.assert_equal(arr_1d, a[:, 0])
    
    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1], 1)
    assert i == 1
    assert bc_idx == (1,)
    np.testing.assert_equal(arr_1d, a[1, :])

    # 3d
    a = np.arange(8).reshape(2, 2, 2)
    arr_1d, i, bc_idx = _get_arr_1d(a, [0, 0, 0], 0)
    assert i == 0
    assert bc_idx == (0, 0)
    np.testing.assert_equal(arr_1d, a[:, 0, 0])
    
    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1, 0], 1)
    assert i == 1
    assert bc_idx == (1, 0)
    np.testing.assert_equal(arr_1d, a[1, :, 0])
    
    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1, 0], 2)
    assert i == 0
    assert bc_idx == (1, 1)
    np.testing.assert_equal(arr_1d, a[1, 1, :])



def test_individual_boundaries():
    """ test setting individual boundaries """
    g = UnitGrid([2])
    for data in ['value', {'value': 1}, {'type': 'value', 'value': 1},
                 'derivative', {'derivative': 1},
                 {'type': 'derivative', 'value': 1}, {'mixed': 1},
                 {'type': 'mixed', 'value': 1}, 'extrapolate']:
        bc = BCBase.from_data(g, 0, upper=True, data=data)
        
        assert bc.check_value_rank(0)
        if bc.value == 0:
            assert bc.check_value_rank(1)
        else:
            with pytest.raises(RuntimeError):
                bc.check_value_rank(1)
                
        assert bc == BCBase.from_data(g, 0, upper=True, data=bc)
        assert bc == bc.copy()
        assert isinstance(str(bc), str)
        assert isinstance(repr(bc), str)
        
    # multidimensional
    g2 = UnitGrid([2, 3])
    bc = BCBase.from_data(g2, 0, True, {'type': 'value', 'value': [1, 2]})
    assert bc.check_value_rank(1)
    with pytest.raises(RuntimeError):
        bc.check_value_rank(0)
    assert bc.extract_component(0).value == 1
    assert bc.extract_component(1).value == 2
    
    
        
def test_virtual_points():
    """ test the calculation of virtual points """
    g = UnitGrid([2])
    data = np.array([1, 2])
    
    # test constant boundary conditions
    bc = BCBase.from_data(g, 0, False, {'type': 'value', 'value': 1})
    assert bc.get_virtual_point(data) == pytest.approx(1)
    bc = BCBase.from_data(g, 0, True, {'type': 'value', 'value': 1})
    assert bc.get_virtual_point(data) == pytest.approx(0)
    
    # test derivative boundary conditions (wrt to outwards derivative)
    bc = BCBase.from_data(g, 0, False, {'type': 'derivative', 'value': -1})
    assert bc.get_virtual_point(data) == pytest.approx(0)
    bc = BCBase.from_data(g, 0, True, {'type': 'derivative', 'value': 1})
    assert bc.get_virtual_point(data) == pytest.approx(3)
    
    # test extrapolation
    bc = BCBase.from_data(g, 0, False, 'extrapolate')
    assert bc.get_virtual_point(data) == pytest.approx(0)
    bc = BCBase.from_data(g, 0, True, 'extrapolate')
    assert bc.get_virtual_point(data) == pytest.approx(3)
    
    # test mixed condition
    bc = BCBase.from_data(g, 0, False,
                          {'type': 'mixed', 'value': 4, 'const': 1})
    assert bc.get_virtual_point(data) == pytest.approx(0)
    bc = BCBase.from_data(g, 0, True,
                          {'type': 'mixed', 'value': 2, 'const': 4})
    assert bc.get_virtual_point(data) == pytest.approx(2)
    
    # test curvature for y = 4 * x**2
    data = np.array([1, 9])
    bc = BCBase.from_data(g, 0, False, {'type': 'curvature', 'value': 8})
    assert bc.get_virtual_point(data) == pytest.approx(1)
    bc = BCBase.from_data(g, 0, True, {'type': 'curvature', 'value': 8})
    assert bc.get_virtual_point(data) == pytest.approx(25)

       
        
def test_mixed_condition():
    """ test the calculation of virtual points """
    g = UnitGrid([2])
    data = np.array([1, 2])
    
    for upper in [True, False]:
        bc1 = BCBase.from_data(g, 0, upper,
                               {'type': 'mixed', 'value': 0, 'const': 2})
        bc2 = BCBase.from_data(g, 0, upper, {'derivative': 2})
        assert bc1.get_virtual_point(data) == \
                pytest.approx(bc2.get_virtual_point(data))
                
    bc = BCBase.from_data(g, 0, False, {'type': 'mixed', 'value': np.inf})
    assert bc.get_virtual_point(data) == pytest.approx(-1)
    bc = BCBase.from_data(g, 0, True, {'type': 'mixed', 'value': np.inf})
    assert bc.get_virtual_point(data) == pytest.approx(-2)
        
