"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde.tools.numba import Counter, flat_idx, numba_environment


def test_environment():
    """test function signature checks"""
    assert isinstance(numba_environment(), dict)


def test_flat_idx():
    """test flat_idx function"""
    assert flat_idx(2, 1) == 2
    assert flat_idx(np.arange(2), 1) == 1
    assert flat_idx(np.arange(4).reshape(2, 2), 1) == 1


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
