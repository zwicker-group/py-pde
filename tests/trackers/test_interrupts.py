"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.trackers.interrupts import (
    ConstantInterrupts,
    FixedInterrupts,
    LogarithmicInterrupts,
    RealtimeInterrupts,
    interval_to_interrupts,
)


def test_interrupt_constant():
    """test the ConstantInterrupts class"""
    ival1 = ConstantInterrupts(2)
    ival2 = ival1.copy()  # test copying too

    assert ival1.initialize(1) == pytest.approx(1)
    assert ival1.next(3) == pytest.approx(3)
    assert ival1.next(3) == pytest.approx(5)
    assert ival1.dt == 2

    assert ival2.initialize(0) == pytest.approx(0)
    assert ival2.next(3) == pytest.approx(4)
    assert ival2.next(3) == pytest.approx(6)
    assert ival2.dt == 2

    ival3 = ival1.copy()  # test copying after starting too
    assert ival3.initialize(0) == pytest.approx(0)
    assert ival3.next(3) == pytest.approx(4)
    assert ival3.next(3) == pytest.approx(6)
    assert ival3.dt == 2

    ival = interval_to_interrupts(2)
    assert ival.initialize(1) == pytest.approx(1)
    assert ival.next(3) == pytest.approx(3)
    assert ival.next(3) == pytest.approx(5)
    assert ival.dt == 2


def test_interrupt_tstart():
    """test the t_start argument of interrupts"""
    ival = ConstantInterrupts(dt=2, t_start=7)
    assert ival.initialize(0) == pytest.approx(7)
    assert ival.next(3) == pytest.approx(9)
    assert ival.next(3) == pytest.approx(11)
    assert ival.next(3) == pytest.approx(13)
    assert ival.dt == 2


def test_interrupt_logarithmic():
    """test the LogarithmicInterrupts class"""
    ival = LogarithmicInterrupts(2, factor=2)
    assert ival.initialize(0) == pytest.approx(0)
    assert ival.dt == 1
    assert ival.next(3) == pytest.approx(4)
    assert ival.dt == 2
    assert ival.next(3) == pytest.approx(8)
    assert ival.dt == 4
    assert ival.next(3) == pytest.approx(16)
    assert ival.dt == 8


def test_interrupt_realtime():
    """test the RealtimeInterrupts class"""
    for ival in [RealtimeInterrupts("0:01"), interval_to_interrupts("0:01")]:
        assert ival.initialize(0) == pytest.approx(0)
        i1, i2, i3 = ival.next(1), ival.next(1), ival.next(1)
        assert i3 > i2 > i1 > 0


def test_interrupt_fixed():
    """test the FixedInterrupts class"""
    ival = FixedInterrupts([1, 3])
    assert ival.initialize(0) == pytest.approx(1)
    assert ival.dt == 1
    assert ival.next(1) == pytest.approx(3)
    assert ival.dt == 2
    assert np.isinf(ival.next(1))

    ival = FixedInterrupts([1, 3, 5])
    assert ival.initialize(2) == pytest.approx(3)
    assert ival.dt == 1
    assert ival.next(4) == pytest.approx(5)
    assert ival.dt == 2
    assert np.isinf(ival.next(1))

    ival = FixedInterrupts([1, 3, 5, 7])
    assert ival.initialize(0) == pytest.approx(1)
    assert ival.dt == 1
    assert ival.next(6) == pytest.approx(7)
    assert ival.dt == 6

    ival = interval_to_interrupts([1, 3])
    assert np.isinf(ival.initialize(4))

    ival = interval_to_interrupts(np.arange(3))
    assert ival.initialize(0) == pytest.approx(0)
    assert ival.dt == 0
    assert ival.next(0) == pytest.approx(1)
    assert ival.dt == 1
    assert ival.next(0) == pytest.approx(2)
    assert ival.dt == 1
    assert np.isinf(ival.next(0))

    # edge cases
    ival = FixedInterrupts([])
    assert np.isinf(ival.initialize(0))
    ival = FixedInterrupts(1)
    assert ival.initialize(0) == pytest.approx(1)
    assert np.isinf(ival.next(0))
    with pytest.raises(AssertionError):
        ival = FixedInterrupts([[1]])
