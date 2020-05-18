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
    assert bc_idx == (...,)
    np.testing.assert_equal(arr_1d, a)

    # 2d
    a = np.arange(4).reshape(2, 2)
    arr_1d, i, bc_idx = _get_arr_1d(a, [0, 0], 0)
    assert i == 0
    assert bc_idx == (..., 0)
    np.testing.assert_equal(arr_1d, a[:, 0])
    
    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1], 1)
    assert i == 1
    assert bc_idx == (..., 1)
    np.testing.assert_equal(arr_1d, a[1, :])

    # 3d
    a = np.arange(8).reshape(2, 2, 2)
    arr_1d, i, bc_idx = _get_arr_1d(a, [0, 0, 0], 0)
    assert i == 0
    assert bc_idx == (..., 0, 0)
    np.testing.assert_equal(arr_1d, a[:, 0, 0])
    
    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1, 0], 1)
    assert i == 1
    assert bc_idx == (..., 1, 0)
    np.testing.assert_equal(arr_1d, a[1, :, 0])
    
    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1, 0], 2)
    assert i == 0
    assert bc_idx == (..., 1, 1)
    np.testing.assert_equal(arr_1d, a[1, 1, :])



def test_individual_boundaries():
    """ test setting individual boundaries """
    g = UnitGrid([2])
    for data in ['value', {'value': 1}, {'type': 'value', 'value': 1},
                 'natural', 'derivative', {'derivative': 1},
                 {'type': 'derivative', 'value': 1}, {'mixed': 1},
                 {'type': 'mixed', 'value': 1}, 'extrapolate']:
        bc = BCBase.from_data(g, 0, upper=True, data=data, rank=0)
        
        assert isinstance(str(bc), str)
        assert isinstance(repr(bc), str)
        assert bc.rank == 0
        assert bc.homogeneous
        bc.check_value_rank(0)
        with pytest.raises(RuntimeError):
            bc.check_value_rank(1)

        for bc_copy in [BCBase.from_data(g, 0, upper=True, data=bc, rank=0),
                        bc.copy()]:
            assert bc == bc_copy
            assert bc._cache_hash() == bc_copy._cache_hash()

    assert bc.extract_component() == bc
        


def test_individual_boundaries_multidimensional():
    """ test setting individual boundaries in 2d """
    g2 = UnitGrid([2, 3])
    bc = BCBase.from_data(g2, 0, True, {'type': 'value', 'value': [1, 2]},
                          rank=1)

    assert isinstance(str(bc), str)
    assert isinstance(repr(bc), str)
    assert bc.rank == 1
    assert bc.homogeneous
    bc.check_value_rank(1)
    with pytest.raises(RuntimeError):
        bc.check_value_rank(0)
        
    bc_comp = bc.extract_component(0)
    assert bc_comp.rank == 0
    assert bc_comp.value == 1
    bc_comp = bc.extract_component(1)
    assert bc_comp.rank == 0
    assert bc_comp.value == 2
    
    for bc_copy in [BCBase.from_data(g2, 0, upper=True, data=bc, rank=1),
                    bc.copy()]:
        assert bc == bc_copy
        assert bc._cache_hash() == bc_copy._cache_hash()

    
        
def test_virtual_points():
    """ test the calculation of virtual points """
    g = UnitGrid([2])
    data = np.array([1, 2])
    
    # test constant boundary conditions
    bc = BCBase.from_data(g, 0, False, {'type': 'value', 'value': 1})
    assert bc.get_virtual_point(data) == pytest.approx(1)
    assert not bc.value_is_linked
    bc = BCBase.from_data(g, 0, True, {'type': 'value', 'value': 1})
    assert bc.get_virtual_point(data) == pytest.approx(0)
    assert not bc.value_is_linked
    
    # test derivative boundary conditions (wrt to outwards derivative)
    for up, b, val in [(False, {'type': 'derivative', 'value': -1}, 0),
                        (True, {'type': 'derivative', 'value': 1}, 3),
                        (False, 'extrapolate', 0),
                        (True, 'extrapolate', 3),
                        (False, {'type': 'mixed', 'value': 4, 'const': 1}, 0),
                        (True, {'type': 'mixed', 'value': 2, 'const': 4}, 2)]:
        bc = BCBase.from_data(g, 0, up, b)
        assert bc.upper == up
        assert bc.get_virtual_point(data) == pytest.approx(val)
        assert not bc.value_is_linked
        ev = bc.make_virtual_point_evaluator()
        assert ev(data, (2,) if up else (-1,)) == pytest.approx(val)
    
    # test curvature for y = 4 * x**2
    data = np.array([1, 9])
    bc = BCBase.from_data(g, 0, False, {'type': 'curvature', 'value': 8})
    assert bc.get_virtual_point(data) == pytest.approx(1)
    assert not bc.value_is_linked
    bc = BCBase.from_data(g, 0, True, {'type': 'curvature', 'value': 8})
    assert bc.get_virtual_point(data) == pytest.approx(25)
    assert not bc.value_is_linked

       
       
@pytest.mark.parametrize('upper', [False, True])
def test_virtual_points_linked_data(upper):
    """ test the calculation of virtual points with linked_data """
    g = UnitGrid([2, 2])
    point = (1, 1) if upper else (0, 0)
    data = np.zeros(g.shape)
    
    # test constant boundary conditions
    bc_data = np.array([1, 1])
    bc = BCBase.from_data(g, 0, upper, {'type': 'value', 'value': bc_data})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 3

    assert bc.get_virtual_point(data, point) == pytest.approx(6)
    ev = bc.make_virtual_point_evaluator()
    assert ev(data, point) == pytest.approx(6)

    # test derivative boundary conditions (wrt to outwards derivative)
    bc = BCBase.from_data(g, 0, upper, {'type': 'derivative', 'value': bc_data})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 4

    assert bc.get_virtual_point(data, point) == pytest.approx(4)
    ev = bc.make_virtual_point_evaluator()
    assert ev(data, point) == pytest.approx(4)
        
    # test derivative boundary conditions (wrt to outwards derivative)
    bc = BCBase.from_data(g, 0, upper, {'type': 'mixed', 'value': bc_data,
                                        'const': 3})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 4

    assert bc.get_virtual_point(data, point) == pytest.approx(1)
    ev = bc.make_virtual_point_evaluator()
    assert ev(data, point) == pytest.approx(1)
        
        
        
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
    
    g = UnitGrid([2, 2])
    data = np.ones(g.shape)
    bc = BCBase.from_data(g, 0, False, {'type': 'mixed', 'value': [1, 2],
                                        'const': [3, 4]})
    assert bc.get_virtual_point(data, (0, 0)) == pytest.approx(2 + 1/3)
    assert bc.get_virtual_point(data, (0, 1)) == pytest.approx(2)
        
        
        
def test_inhomogeneous_bcs():
    """ test inhomogeneous boundary conditions """
    g = UnitGrid([2, 2])
    data = np.ones((2, 2))
    
    # first order bc
    bc_x = BCBase.from_data(g, 0, True, {'value': 'y'})
    assert isinstance(str(bc_x), str)
    assert bc_x.rank == 0
    assert bc_x.get_virtual_point(data, (1, 0)) == pytest.approx(0)
    assert bc_x.get_virtual_point(data, (1, 1)) == pytest.approx(2)

    # second order bc
    bc_x = BCBase.from_data(g, 0, True, {'curvature': 'y'})
    assert isinstance(str(bc_x), str)
    assert bc_x.rank == 0
    assert bc_x.get_virtual_point(data, (1, 0)) == pytest.approx(1.5)
    assert bc_x.get_virtual_point(data, (1, 1)) == pytest.approx(2.5)
    
    ev = bc_x.make_virtual_point_evaluator()
    assert ev(data, (1, 0)) == pytest.approx(1.5)
    assert ev(data, (1, 1)) == pytest.approx(2.5)

    ev = bc_x.make_adjacent_evaluator()
    assert ev(*_get_arr_1d(data, (0, 0), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (0, 1), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (1, 0), axis=0)) == pytest.approx(1.5)
    assert ev(*_get_arr_1d(data, (1, 1), axis=0)) == pytest.approx(2.5)
    # test lower bc
    bc_x = BCBase.from_data(g, 0, False, {'curvature': 'y'})
    ev = bc_x.make_adjacent_evaluator()
    assert ev(*_get_arr_1d(data, (1, 0), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (1, 1), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (0, 0), axis=0)) == pytest.approx(1.5)
    assert ev(*_get_arr_1d(data, (0, 1), axis=0)) == pytest.approx(2.5)
