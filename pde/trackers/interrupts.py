"""Module defining classes for time interrupts for trackers.

The provided interrupt classes are:

.. autosummary::
   :nosignatures:

   ConstantInterrupts
   FixedInterrupts
   LogarithmicInterrupts
   GeometricInterrupts
   RealtimeInterrupts
   parse_interrupt

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import copy
import math
import re
import time
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional, TypeVar, Union

import numpy as np

from ..tools.parse_duration import parse_duration

InfoDict = Optional[dict[str, Any]]
TInterrupt = TypeVar("TInterrupt", bound="InterruptsBase")


class InterruptsBase(metaclass=ABCMeta):
    """Base class for implementing interrupts."""

    dt: float
    """float: current time difference between interrupts"""

    def copy(self):
        return copy.copy(self)

    @abstractmethod
    def initialize(self, t: float) -> float:
        """Initialize the interrupt class.

        Args:
            t (float): The starting time of the simulation

        Returns:
            float: The first time the simulation needs to be interrupted
        """

    @abstractmethod
    def next(self, t: float) -> float:
        """Computes the next time point.

        Args:
            t (float):
                The current time point of the simulation. The returned next time point
                lies later than this time, so interrupts might be skipped.

        Returns:
            float: The next time point
        """


class FixedInterrupts(InterruptsBase):
    """Interrupts at fixed, predetermined times."""

    def __init__(self, interrupts: np.ndarray | Sequence[float]):
        self.interrupts = np.atleast_1d(interrupts)
        if self.interrupts.ndim != 1:
            raise ValueError("`interrupts` must be a 1d sequence")

    def __repr__(self):
        return f"{self.__class__.__name__}(interrupts={self.interrupts})"

    def copy(self):
        return self.__class__(interrupts=self.interrupts.copy())

    def initialize(self, t: float) -> float:
        self._index = -1
        return self.next(t)

    def next(self, t: float) -> float:
        try:
            # Determine time of the last interrupt. This value does not make much sense
            # for the first interrupt, so we simply use the current time
            if self._index < 0:
                t_last = t
            else:
                t_last = self.interrupts[self._index]

            # fetch the next entry that is after the current time `t`
            self._index += 1
            t_next: float = self.interrupts[self._index]  # fetch next time point
            while t_next < t:  # ensure time point lies in the future
                self._index += 1
                t_next = self.interrupts[self._index]

            self.dt = t_next - t_last
            return t_next

        except IndexError:
            # iterator has been exhausted -> never break again
            return math.inf


class ConstantInterrupts(InterruptsBase):
    """Interrupts equidistantly spaced in time."""

    def __init__(self, dt: float = 1, t_start: float | None = None):
        """
        Args:
            dt (float):
                The duration between subsequent interrupts. This is measured in
                simulation time units.
            t_start (float, optional):
                The time after which the tracker becomes active. If omitted, the tracker
                starts recording right away. This argument can be used for an initial
                equilibration period during which no data is recorded.
        """
        self.dt = float(dt)
        self.t_start = None if t_start is None else float(t_start)
        self._t_next: float | None = None  # next time it should be called

    def __repr__(self):
        return f"{self.__class__.__name__}(dt={self.dt:g}, t_start={self.t_start})"

    def initialize(self, t: float) -> float:
        if self.t_start is None:
            self._t_next = t
        else:
            self._t_next = max(t, self.t_start)
        return self._t_next

    def next(self, t: float) -> float:
        # move next interrupt time by the appropriate interrupt
        self._t_next += self.dt  # type: ignore

        # make sure that the new interrupt time is in the future
        if self._t_next <= t:
            # add `dt` until _t_next is in the future (larger than t)
            n = math.ceil((t - self._t_next) / self.dt)
            self._t_next += self.dt * n
            # adjust in special cases where float-point math fails us
            if self._t_next < t:
                self._t_next += self.dt

        return self._t_next


class LogarithmicInterrupts(ConstantInterrupts):
    r"""Interrupts with successively increased spacing.

    The durations between interrupts increases by a constant factor :math:`f`:

    .. math::
        t_{i+1} = t_i + \Delta t_i \qquad \text{with} \qquad
        \Delta t_{i+1} = f \Delta t_i

    starting with initial values :math:`t_0` and :math:`\Delta t_0`. This results in
    exponentially spaced interrupts :math:`t_i = a + b f^i`, where
    :math:`a = t_0 - \Delta t_0 (f - 1)^{-1}` and :math:`b = \Delta t_0 (f - 1)^{-1}`.

    Note that the geometric sequence described above can be disrupted if other
    interrupts interfere. This class ensures ever increasing durations between its
    interrupts, at the cost of potentially oddly spaced times. If a geometric sequence
    is required, use :class:`GeometricInterrupts` instead.
    """

    def __init__(
        self, dt_initial: float = 1, factor: float = 1, t_start: float | None = None
    ):
        r"""
        Args:
            dt_initial (float):
                The initial duration :math:`\Delta t_0` between subsequent interrupts.
                This is measured in simulation time units.
            factor (float):
                The factor :math:`f` by which the time between interrupts is increased
                after every interrupt. Values larger than one lead to time interrupts
                that are increasingly further apart.
            t_start (float, optional):
                The time :math:`t_0` after which the tracker becomes active. If omitted,
                the tracker starts recording right away. This argument can be used for
                an initial equilibration period during which no data is recorded.
        """
        # convert arguments
        self.dt_initial = float(dt_initial)
        self.factor = float(factor)

        # initialize instance
        super().__init__(dt=self.dt_initial / self.factor, t_start=t_start)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dt={self.dt_initial:g}, "
            f"factor={self.factor:g}, t_start={self.t_start})"
        )

    def value(self, iteration: int) -> float:
        """Calculate value of i-th interrupt.

        Args:
            iteration (int):
                The iteration of the interrupt

        Returns:
            float: time of the i-th interrupt
        """
        t_start = 0 if self.t_start is None else self.t_start
        a = t_start - self.dt_initial / (self.factor - 1)
        b = self.dt_initial / (self.factor - 1)
        return a + b * self.factor**iteration

    def next(self, t: float) -> float:
        self.dt *= self.factor
        return super().next(t)


class GeometricInterrupts(InterruptsBase):
    r"""Interrupts from the geometric sequence :math:`t_i = \Delta t f^i`

    In contrast to :class:`LogarithmicInterrupts`, this class ensures that time points
    lie on the geometric sequence given above. However, data points might be skipped if
    the simulations progress too quickly.
    """

    def __init__(self, scale: float, factor: float):
        r"""
        Args:
            scale (float):
                Time scale :math:`\Delta t`.
            factor (float):
                Scale factor :math:`f`.
        """
        self.scale = float(scale)
        self.factor = float(factor)
        if factor <= 0:
            raise ValueError("Factor must be a positive number")
        self._t_next: float | None = None  # next time it should be called

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:g}, factor={self.factor:g})"
        )

    def value(self, iteration: int) -> float:
        """Calculate value of i-th interrupt.

        Args:
            iteration (int):
                The iteration of the interrupt

        Returns:
            float: time of the i-th interrupt
        """
        return self.scale * self.factor**iteration

    def initialize(self, t: float) -> float:
        return self.next(t)

    def next(self, t: float) -> float:
        # determine minimal time we need to return
        if self._t_next is None:
            # get a time slightly below the first interrupt
            t_min = self.scale * self.factor**-0.5
        else:
            # get a time slightly above the last returned interrupt
            t_min = self._t_next * self.factor**0.5
        # current time might have surpassed the estimate above
        t_min = max(t, t_min)
        # estimate (fractional) iteration number of current time point
        i = np.log(t_min / self.scale) / np.log(self.factor)
        # round up the fractional estimate and get associated interrupt time
        self._t_next = self.scale * self.factor ** np.ceil(i)
        return self._t_next


class RealtimeInterrupts(ConstantInterrupts):
    """Interrupts spaced equidistantly in real time.

    This spacing is only achieved approximately and depends on the initial value
    set by `dt_initial` and the actual variation in computation speed.
    """

    def __init__(self, duration: float | str, dt_initial: float = 0.01):
        """
        Args:
            duration (float or str):
                The duration (in real seconds) that the interrupts should be spaced
                apart. The duration can also be given as a string, which is then parsed
                using the function :func:`~pde.tools.parse_duration.parse_duration`.
            dt_initial (float):
                The initial duration between subsequent interrupts. This is measured in
                simulation time units.
        """
        super().__init__(dt=dt_initial)
        try:
            self.duration = float(duration)
        except Exception:
            td = parse_duration(str(duration))
            self.duration = td.total_seconds()
        self._last_time: float | None = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(duration={self.duration:g}, "
            f"dt_initial={self.dt:g})"
        )

    def initialize(self, t: float) -> float:
        self._last_time = time.monotonic()
        return super().initialize(t)

    def next(self, t: float) -> float:
        if self._last_time is None:
            self._last_time = time.monotonic()
        else:
            # adapt time step
            current_time = time.monotonic()
            time_passed = current_time - self._last_time
            if time_passed > 0:
                # predict new time step, but limit it from below, to avoid problems with
                # simulations where a single step takes a long time
                dt_predict = max(1e-3, self.dt * self.duration / time_passed)
                # use geometric average to provide some smoothing
                self.dt = math.sqrt(self.dt * dt_predict)
            else:
                self.dt *= 2
            self._last_time = current_time
        return super().next(t)


InterruptData = Union[InterruptsBase, int, float, str, Sequence[float], np.ndarray]


def parse_interrupt(data: InterruptData) -> InterruptsBase:
    """Create interrupt class from various data formats.

    Args:
        data (str or number or :class:`InterruptsBase`):
            Data determining the interrupt class. If this is a :class:`InterruptsBase`,
            it is simply returned, numbers imply :class:`ConstantInterrupts`, a string
            is generally parsed as a time for :class:`RealtimeInterrupts`, and lists are
            interpreted as :class:`FixedInterrupts`. Instance of
            :class:`GeometricInterrupts` can be constructed with the special string
            :code:`"geometric(SCALE, FACTOR)"`, specifying the `scale` and `factor`
            values directly as numbers.

    Returns:
        :class:`InterruptsBase`: An instance that represents the interrupt
    """
    if isinstance(data, InterruptsBase):
        # is already the correct class
        return data

    elif isinstance(data, (int, float)):
        # is a number, so we assume a constant interrupt of that duration
        return ConstantInterrupts(data)

    elif isinstance(data, str):
        # a string is either a special geometric sequence of a time duration
        if data.startswith("geometric"):
            regex = r"geometric\(\s*([0-9.e+-]*)\s*,\s*([0-9.e+-]*)\s*\)"
            matches = re.search(regex, data, re.IGNORECASE)
            if matches:
                scale = float(matches.group(1))
                factor = float(matches.group(2))
                return GeometricInterrupts(scale, factor)
            else:
                raise ValueError(f"Could not interpret `{data}` as interrupt")
        else:
            return RealtimeInterrupts(data)

    elif hasattr(data, "__iter__"):
        # a sequence is supposed to give fixed time points for interrupts
        return FixedInterrupts(data)

    else:
        # anything else we cannot handle
        raise TypeError(f"Cannot parse interrupt data `{data}`")
