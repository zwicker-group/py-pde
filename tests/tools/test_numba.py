"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.numba import (
    Counter,
    flat_idx,
    jit,
    make_array_constructor,
    numba_environment,
)


def test_environment():
    """Test function signature checks."""
    assert isinstance(numba_environment(), dict)


def test_flat_idx():
    """Test flat_idx function."""
    # testing the numpy version
    assert flat_idx(2, 1) == 2
    assert flat_idx(np.arange(2), 1) == 1
    assert flat_idx(np.arange(4).reshape(2, 2), 1) == 1

    # testing the numba compiled version
    @jit
    def get_sparse_matrix_data(data):
        return flat_idx(data, 1)

    assert get_sparse_matrix_data(2) == 2
    assert get_sparse_matrix_data(np.arange(2)) == 1
    assert get_sparse_matrix_data(np.arange(4).reshape(2, 2)) == 1


def test_counter():
    """Test Counter implementation."""
    c1 = Counter()
    assert int(c1) == 0
    assert c1 == 0
    assert str(c1) == "0"

    c1.increment()
    assert int(c1) == 1

    c1 += 2
    assert int(c1) == 3

    c2 = Counter(3)
    assert c1 is not c2
    assert c1 == c2


@pytest.mark.parametrize(
    "arr", [np.arange(5), np.linspace(0, 1, 3), np.arange(12).reshape(3, 4)[1:, 2:]]
)
def test_make_array_constructor(arr):
    """Test implementation to create array."""
    constructor = jit(make_array_constructor(arr))
    arr2 = constructor()
    np.testing.assert_equal(arr, arr2)
    assert np.shares_memory(arr, arr2)
