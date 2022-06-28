"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np

from pde.tools.numba import (
    JIT_COUNT,
    Counter,
    flat_idx,
    jit_allocate_out,
    numba_environment,
)


def test_environment():
    """test function signature checks"""
    assert isinstance(numba_environment(), dict)


def test_flat_idx():
    """test flat_idx function"""
    assert flat_idx(2, 1) == 2
    assert flat_idx(np.arange(2), 1) == 1
    assert flat_idx(np.arange(4).reshape(2, 2), 1) == 1


def test_jit_allocate_out_1arg():
    """test jit_allocate_out of functions with 1 argument"""

    def f(arr, out=None, args=None):
        out[:] = arr
        return out

    jit_count = int(JIT_COUNT)
    a = np.linspace(0, 1, 3)
    g = jit_allocate_out(out_shape=a.shape)(f)
    np.testing.assert_equal(g(a), a)
    np.testing.assert_equal(jit_allocate_out(f)(a), a)
    if nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) == jit_count
    else:
        assert int(JIT_COUNT) == jit_count + 2


def test_jit_allocate_out_2arg():
    """test jit_allocate_out of functions with 1 argument"""

    def f(a, b, out=None, args=None):
        out[:] = a + b
        return out

    jit_count = int(JIT_COUNT)
    a = np.linspace(0, 1, 3)
    b = np.linspace(1, 2, 3)
    c = np.linspace(1, 3, 3)
    g = jit_allocate_out(out_shape=a.shape, num_args=2)(f)
    np.testing.assert_equal(g(a, b), c)
    np.testing.assert_equal(jit_allocate_out(num_args=2)(f)(a, b), c)
    if nb.config.DISABLE_JIT:
        assert int(JIT_COUNT) == jit_count
    else:
        assert int(JIT_COUNT) == jit_count + 2


def test_counter():
    """test Counter implementation"""
    c1 = Counter()
    assert int(c1) is 0
    assert c1 == 0
    assert str(c1) == "0"

    c1.increment()
    assert int(c1) is 1

    c1 += 2
    assert int(c1) is 3

    c2 = Counter(3)
    assert c1 is not c2
    assert c1 == c2
