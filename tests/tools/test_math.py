"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.math import OnlineStatistics, SmoothData1D


def test_SmoothData1D(rng):
    """test smoothing"""
    x = rng.uniform(0, 1, 500)
    xs = np.linspace(0, 1, 16)[1:-1]

    s = SmoothData1D(x, np.ones_like(x), 0.1)
    np.testing.assert_allclose(s(xs), 1)
    np.testing.assert_allclose(s.derivative(xs), 0, atol=1e-14)

    s = SmoothData1D(x, x, 0.02)
    np.testing.assert_allclose(s(xs), xs, atol=0.1)
    np.testing.assert_allclose(s.derivative(xs), 1, atol=0.3)

    s = SmoothData1D(x, np.sin(x), 0.05)
    np.testing.assert_allclose(s(xs), np.sin(xs), atol=0.1)
    np.testing.assert_allclose(s.derivative(xs)[1:-1], np.cos(xs)[1:-1], atol=0.3)

    assert -0.1 not in s
    assert x.min() in s
    assert 0.5 in s
    assert x.max() in s
    assert 1.1 not in s

    x = np.arange(3)
    y = [0, 1, np.nan]
    s = SmoothData1D(x, y)
    assert s(0.5) == pytest.approx(0.5)


def test_online_statistics():
    """test OnlineStatistics class"""
    stat = OnlineStatistics()

    stat.add(1)
    stat.add(2)

    assert stat.mean == pytest.approx(1.5)
    assert stat.std == pytest.approx(0.5)
    assert stat.var == pytest.approx(0.25)
    assert stat.count == 2
    assert stat.to_dict() == pytest.approx(
        {"min": 1, "max": 2, "mean": 1.5, "std": 0.5, "count": 2}
    )

    stat = OnlineStatistics()
    assert stat.to_dict()["count"] == 0
