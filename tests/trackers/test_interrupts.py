"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

import pde
from pde.trackers.interrupts import (
    ConstantInterrupts,
    FixedInterrupts,
    GeometricInterrupts,
    LogarithmicInterrupts,
    RealtimeInterrupts,
    parse_interrupt,
)


def test_interrupt_constant():
    """Test the ConstantInterrupts class."""
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

    ival = parse_interrupt(2)
    assert ival.initialize(1) == pytest.approx(1)
    assert ival.next(3) == pytest.approx(3)
    assert ival.next(3) == pytest.approx(5)
    assert ival.dt == 2


def test_interrupt_tstart():
    """Test the t_start argument of interrupts."""
    ival = ConstantInterrupts(dt=2, t_start=7)
    assert ival.initialize(0) == pytest.approx(7)
    assert ival.next(3) == pytest.approx(9)
    assert ival.next(3) == pytest.approx(11)
    assert ival.next(3) == pytest.approx(13)
    assert ival.dt == 2


def test_interrupt_logarithmic():
    """Test the LogarithmicInterrupts class."""
    ival = LogarithmicInterrupts(2, factor=2)
    assert ival.initialize(0) == pytest.approx(0)
    assert ival.dt == 1
    assert ival.next(3) == pytest.approx(4)
    assert ival.dt == 2
    assert ival.next(3) == pytest.approx(8)
    assert ival.dt == 4
    assert ival.next(3) == pytest.approx(16)
    assert ival.dt == 8


def test_interrupt_logarithmic_seq(rng):
    """Test the LogarithmicInterrupts class."""
    dt, factor, t_start = rng.uniform(1, 2, size=3)
    ival = LogarithmicInterrupts(dt, factor=factor, t_start=t_start)

    ts0 = [ival.initialize(0)]
    ts0 += [ival.next(0) for _ in range(8)]
    ts1 = [ival.value(i) for i in range(9)]
    np.testing.assert_almost_equal(ts0, ts1)

    ts2 = t_start + (1 - factor ** np.arange(9)) / (1 - factor) * dt
    np.testing.assert_almost_equal(ts0, ts2)


def test_interrupt_geometric(rng):
    """Test the GeometricInterrupts class."""
    scale, factor = rng.uniform(1, 2, size=2)
    ival = GeometricInterrupts(scale, factor)

    ts0 = [ival.initialize(0)]
    ts0 += [ival.next(0) for _ in range(8)]
    ts1 = [ival.value(i) for i in range(9)]
    np.testing.assert_almost_equal(ts0, ts1)

    ts2 = scale * factor ** np.arange(9)
    np.testing.assert_almost_equal(ts0, ts2)

    with pytest.raises(ValueError):
        GeometricInterrupts(10, -1)

    interrupt = parse_interrupt("geometric(10, 1.1)")
    assert isinstance(interrupt, GeometricInterrupts)
    assert interrupt.scale == 10
    assert interrupt.factor == 1.1

    interrupt = parse_interrupt("geometric(1,2)")
    assert isinstance(interrupt, GeometricInterrupts)
    assert interrupt.scale == 1
    assert interrupt.factor == 2

    interrupt = parse_interrupt("geometric( .1e-4 , 2.E2 )")
    assert isinstance(interrupt, GeometricInterrupts)
    assert interrupt.scale == 0.1e-4
    assert interrupt.factor == 2.0e2

    with pytest.raises(ValueError):
        parse_interrupt("geometric(1)")
    with pytest.raises(ValueError):
        parse_interrupt("geometric(1,2,3)")
    with pytest.raises(ValueError):
        parse_interrupt("geometric(1,zero)")


def test_interrupt_realtime():
    """Test the RealtimeInterrupts class."""
    for ival in [RealtimeInterrupts("0:01"), parse_interrupt("0:01")]:
        assert ival.initialize(0) == pytest.approx(0)
        i1, i2, i3 = ival.next(1), ival.next(1), ival.next(1)
        assert i3 > i2 > i1 > 0


def test_interrupt_fixed():
    """Test the FixedInterrupts class."""
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

    ival = parse_interrupt([1, 3])
    assert np.isinf(ival.initialize(4))

    ival = parse_interrupt(np.arange(3))
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
    with pytest.raises(ValueError):
        ival = FixedInterrupts([[1]])


@pytest.mark.parametrize("t_start", [-1, 0, 1])
@pytest.mark.parametrize("adaptive", [False, True])
def test_interrupt_integrated(t_start, adaptive):
    """Test how interrupts are used in py-pde."""

    # Initialize the equation and the space.
    state = pde.ScalarField(pde.UnitGrid([1]))

    # Solve the equation and store the trajectory.
    storage = pde.MemoryStorage()
    eq = pde.DiffusionPDE()
    dt = 0.001
    final_state = eq.solve(
        state,
        t_range=[t_start, t_start + 0.25],
        dt=dt,
        adaptive=adaptive,
        backend="numpy",
        tracker=storage.tracker(0.1),
    )

    assert storage.times[0] == pytest.approx(t_start)
    assert storage.times[1] == pytest.approx(t_start + 0.1)
    assert len(storage) == 3
