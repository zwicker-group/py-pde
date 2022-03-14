"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde.trackers import ConstantIntervals, LogarithmicIntervals


def test_intervals():
    """test the interval classes"""
    ival1 = ConstantIntervals(2)
    ival2 = ival1.copy()  # test copying too
    for ival in [ival1, ival2]:
        assert ival.next(3) == pytest.approx(5)
        assert ival.next(3) == pytest.approx(7)
        assert ival.next(3) == pytest.approx(9)

    ival = LogarithmicIntervals(2, factor=2)
    assert ival.next(3) == pytest.approx(5)
    assert ival.next(3) == pytest.approx(9)
    assert ival.next(3) == pytest.approx(17)


def test_interval_tstart():
    """test the t_start argument of intervals"""
    ival = ConstantIntervals(dt=2, t_start=7)
    assert ival.next(3) == pytest.approx(9)
    assert ival.next(3) == pytest.approx(11)
    assert ival.next(3) == pytest.approx(13)
