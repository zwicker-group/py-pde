"""
Module defining classes for time intervals for trackers

The provided interval classes are:

.. autosummary::
   :nosignatures:

   ConstantIntervals
   LogarithmicIntervals
   RealtimeIntervals
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

import copy
import math
import time
from typing import Optional, Union, Dict, Any

from ..tools.parse_duration import parse_duration



Real = Union[float, int]
InfoDict = Optional[Dict[str, Any]]



class ConstantIntervals():
    """ class representing equidistantly spaced time intervals """
    
    def __init__(self, dt: float = 1, t_start: Optional[float] = None):
        """
        Args:
            dt (float): The duration between subsequent intervals. This is
                measured in simulation time units.
            t_start (float, optional): The time after which the tracker becomes
                active. If omitted, the tracker starts recording right away.
                This argument can be used for an initial equilibration period
                during which no data is recorded.
        """
        self.dt = float(dt)
        self.t_start = t_start
        self._t_next: Optional[float] = None  # next time it should be called
        
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(dt={self.dt:g}, '
                f't_start={self.t_start})')
        
        
    def copy(self):
        """ return a copy of this instance """
        return copy.copy(self)
        
        
    def _initialize(self, t: float) -> float:
        """ initialize the tracker
        
        Args:
            t (float): The starting time of the simulation
                
        Returns:
            float: The first time the tracker needs to handle data
        """
        if self.t_start is None:
            self._t_next = t
        else:
            self._t_next = max(t, self.t_start)
        return self._t_next
        
        
    def next(self, t: float) -> float:
        """ computes the next time point based on the current time t
        
        Args:
            t (float): The current time point of the simulation
        """
        if self._t_next is None:
            self._initialize(t)
            
        self._t_next = self._t_next + self.dt  # type: ignore

        # make sure that the new time `_t_next` is larger than t
        while self._t_next <= t:
            self._t_next += self.dt
        return self._t_next
    


class LogarithmicIntervals(ConstantIntervals):
    """ class representing logarithmically spaced time intervals """
    
    def __init__(self, dt_initial: float = 1, factor: float = 1,
                 t_start: Optional[float] = None):
        """
        Args:
            dt_initial (float): The initial duration between subsequent
                intervals. This is measured in simulation time units.
            factor (float): The factor by which the time between intervals is
                increased every time. Values larger than one lead to time
                intervals that are increasingly further apart.
            t_start (float, optional): The time after which the tracker becomes
                active. If omitted, the tracker starts recording right away.
                This argument can be used for an initial equilibration period
                during which no data is recorded.
        """
        super().__init__(dt=dt_initial / factor, t_start=t_start)
        self.factor = float(factor)


    def __repr__(self):
        return (f'{self.__class__.__name__}(dt={self.dt:g}, '
                f'factor={self.factor:g}, t_start={self.t_start})')


    def next(self, t: float) -> float:
        """ computes the next time point based on the current time t
        
        Args:
            t (float): The current time point of the simulation
        """
        self.dt *= self.factor
        return super().next(t)



class RealtimeIntervals(ConstantIntervals):
    """ class representing time intervals spaced equidistantly in real time
    
    This spacing is only achieved approximately and depends on the initial value
    set by `dt_initial` and the actual variation in computation speed.
    """

    def __init__(self, duration: Union[float, str], dt_initial: float = 0.01):
        """
        Args:
            duration (float or str): The duration (in realtime seconds) that the
                intervals should be spaced apart. The duration can also be given
                as a string, which is then parsed using the function
                :func:`~pde.tools.parse_duration.parse_duration`.
            dt_initial (float): The initial duration between subsequent
                intervals. This is measured in simulation time units.
        """
        super().__init__(dt=dt_initial)
        try:
            self.duration = float(duration)
        except Exception:
            td = parse_duration(str(duration))
            self.duration = td.total_seconds()
        self._last_time: Optional[float] = None
        
        
    def __repr__(self):
        return (f'{self.__class__.__name__}(duration={self.duration:g}, '
                f'dt_initial={self.dt:g})')

        
    def _initialize(self, t: float) -> float:
        """ initialize the tracker
        
        Args:
            t (float): The starting time of the simulation
                
        Returns:
            float: The first time the tracker needs to handle data
        """
        self._last_time = time.time()
        return super()._initialize(t)
        
        
    def next(self, t: float) -> float:
        """ computes the next time point based on the current time t
        
        Args:
            t (float): The current time point of the simulation
        """
        if self._last_time is None:
            self._last_time = time.time()
        else:
            # adapt time step
            current_time = time.time()
            time_passed = current_time - self._last_time
            if time_passed > 0:
                dt_predict = self.dt * self.duration / time_passed
                # use geometric average to provide some smoothing
                self.dt = math.sqrt(self.dt * dt_predict)
            else:
                self.dt *= 2
            self._last_time = current_time
        return super().next(t)



IntervalType = ConstantIntervals
IntervalData = Union[IntervalType, Real, str]



def get_interval(interval: IntervalData) -> IntervalType:
    """ create IntervalType from various data formats
    
    If interval is of type :class:`IntervalType` it is simply returned
    """
    if isinstance(interval, IntervalType):
        return interval
    elif isinstance(interval, (int, float)):
        return ConstantIntervals(interval)
    elif isinstance(interval, str):
        return RealtimeIntervals(interval)
    else:
        raise TypeError(f'Do not understand interval type {interval}')



