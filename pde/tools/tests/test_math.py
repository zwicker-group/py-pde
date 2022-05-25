"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.math import SmoothData1D


def test_SmoothData1D():
    """test smoothing"""
    x = np.random.uniform(0, 1, 128)
    xs = np.linspace(0, 1, 16)[1:-1]

    s = SmoothData1D(x, np.ones_like(x))
    np.testing.assert_allclose(s(xs), 1)

    s = SmoothData1D(x, x)
    np.testing.assert_allclose(s(xs), xs, atol=0.1)

    s = SmoothData1D(x, np.sin(x))
    np.testing.assert_allclose(s(xs), np.sin(xs), atol=0.1)

    assert -0.1 not in s
    assert x.min() in s
    assert 0.5 in s
    assert x.max() in s
    assert 1.1 not in s

    x = np.arange(3)
    y = [0, 1, np.nan]
    s = SmoothData1D(x, y)
    assert s(0.5) == pytest.approx(0.5)
