'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np

from ..numba import numba_environment, flat_idx, jit_allocate_out



def test_environment():
    """ test function signature checks """
    assert isinstance(numba_environment(), dict)
    


def test_flat_idx():
    """ test flat_idx function """
    assert flat_idx(2, 1) == 2
    assert flat_idx(np.arange(2), 1) == 1
    assert flat_idx(np.arange(4).reshape(2, 2), 1) == 1



def test_jit_allocate_out_1arg():
    """ test jit_allocate_out of functions with 1 argument """
    def f(arr, out):
        out[:] = arr
        return out
    
    a = np.linspace(0, 1, 3)
    g = jit_allocate_out(out_shape=a.shape)(f)
    np.testing.assert_equal(g(a), a)
    np.testing.assert_equal(jit_allocate_out(f)(a), a)
    


def test_jit_allocate_out_2arg():
    """ test jit_allocate_out of functions with 1 argument """
    def f(a, b, out):
        out[:] = a + b
        return out
    
    a = np.linspace(0, 1, 3)
    b = np.linspace(1, 2, 3)
    c = np.linspace(1, 3, 3)
    g = jit_allocate_out(out_shape=a.shape, num_args=2)(f)
    np.testing.assert_equal(g(a, b), c)
    np.testing.assert_equal(jit_allocate_out(num_args=2)(f)(a, b), c)
    