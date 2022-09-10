"""
Module defining classes for time interrupts for trackers

The provided interrupt classes are:

.. autosummary::
   :nosignatures:

   FixedInterrupts
   ConstantInterrupts
   LogarithmicInterrupts
   RealtimeInterrupts
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

import copy
import math
import time
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from ..tools.parse_duration import parse_duration

InfoDict = Optional[Dict[str, Any]]


class InterruptsBase(metaclass=ABCMeta):
    """base class for implementing interrupts"""

    dt: float
    """float: current time difference between interrupts"""

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def initialize(self, t: float) -> float:
        pass

    @abstractmethod
    def next(self, t: float) -> float:
        pass


class FixedInterrupts(InterruptsBase):
    """class representing a list of interrupt times"""

    def __init__(self, interrupts: Union[np.ndarray, Sequence[float]]):
        self.interrupts = np.atleast_1d(interrupts)
        assert self.interrupts.ndim == 1

    def __repr__(self):
        return f"{self.__class__.__name__}(interrupts={self.interrupts})"

    def copy(self):
        """return a copy of this instance"""
        return self.__class__(interrupts=self.interrupts.copy())

    def initialize(self, t: float) -> float:
        """initialize the interrupt class

        Args:
            t (float): The starting time of the simulation

        Returns:
            float: The first time the simulation needs to be interrupted
        """
        self._index = -1
        return self.next(t)

    def next(self, t: float) -> float:
        """computes the next time point

        Args:
            t (float):
                The current time point of the simulation. The returned next time point
                lies later than this time, so interrupts might be skipped.
        """
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
    """class representing equidistantly spaced time interrupts"""

    def __init__(self, dt: float = 1, t_start: Optional[float] = None):
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
        self.t_start = t_start
        self._t_next: Optional[float] = None  # next time it should be called

    def __repr__(self):
        return f"{self.__class__.__name__}(dt={self.dt:g}, t_start={self.t_start})"

    def copy(self):
        """return a copy of this instance"""
        return copy.copy(self)

    def initialize(self, t: float) -> float:
        """initialize the interrupt class

        Args:
            t (float): The starting time of the simulation

        Returns:
            float: The first time the simulation needs to be interrupted
        """
        if self.t_start is None:
            self._t_next = t
        else:
            self._t_next = max(t, self.t_start)
        return self._t_next

    def next(self, t: float) -> float:
        """computes the next time point

        Args:
            t (float):
                The current time point of the simulation. The returned next time point
                lies later than this time, so interrupts might be skipped.
        """
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
    """class representing logarithmically spaced time interrupts"""

    def __init__(
        self, dt_initial: float = 1, factor: float = 1, t_start: Optional[float] = None
    ):
        """
        Args:
            dt_initial (float):
                The initial duration between subsequent interrupts. This is measured in
                simulation time units.
            factor (float):
                The factor by which the time between interrupts is increased every time.
                Values larger than one lead to time interrupts that are increasingly
                further apart.
            t_start (float, optional):
                The time after which the tracker becomes active. If omitted, the tracker
                starts recording right away. This argument can be used for an initial
                equilibration period during which no data is recorded.
        """
        super().__init__(dt=dt_initial / factor, t_start=t_start)
        self.factor = float(factor)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dt={self.dt:g}, "
            f"factor={self.factor:g}, t_start={self.t_start})"
        )

    def next(self, t: float) -> float:
        """computes the next time point

        Args:
            t (float):
                The current time point of the simulation. The returned next time point
                lies later than this time, so interrupts might be skipped.
        """
        self.dt *= self.factor
        return super().next(t)


class RealtimeInterrupts(ConstantInterrupts):
    """class representing time interrupts spaced equidistantly in real time

    This spacing is only achieved approximately and depends on the initial value
    set by `dt_initial` and the actual variation in computation speed.
    """

    def __init__(self, duration: Union[float, str], dt_initial: float = 0.01):
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
        self._last_time: Optional[float] = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(duration={self.duration:g}, "
            f"dt_initial={self.dt:g})"
        )

    def initialize(self, t: float) -> float:
        """initialize the interrupt class

        Args:
            t (float): The starting time of the simulation

        Returns:
            float: The first time the simulation needs to be interrupted
        """
        self._last_time = time.monotonic()
        return super().initialize(t)

    def next(self, t: float) -> float:
        """computes the next time point

        Args:
            t (float):
                The current time point of the simulation. The returned next time point
                lies later than this time, so interrupts might be skipped.
        """
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


IntervalData = Union[InterruptsBase, float, str, Sequence[float], np.ndarray]


def interval_to_interrupts(data: IntervalData) -> InterruptsBase:
    """create interrupt class from various data formats specifying time intervals

    Args:
        data (str or number or :class:`InterruptsBase`):
            Data determining the interrupt class. If this is a :class:`InterruptsBase`,
            it is simply returned, numbers imply :class:`ConstantInterrupts`, a string
            is parsed as a time for :class:`RealtimeInterrupts`, and lists are
            interpreted as :class:`FixedInterrupts`.

    Returns:
        :class:`InterruptsBase`: An instance that represents the time intervals
    """
    if isinstance(data, InterruptsBase):
        return data
    elif isinstance(data, (int, float)):
        return ConstantInterrupts(data)
    elif isinstance(data, str):
        return RealtimeInterrupts(data)
    elif hasattr(data, "__iter__"):
        return FixedInterrupts(data)
    else:
        raise TypeError(f"Cannot parse interrupt data `{data}`")
