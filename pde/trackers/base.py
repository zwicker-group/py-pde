"""
Base classes for trackers 
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..fields.base import FieldBase
from ..tools.docstrings import fill_in_docstring
from .intervals import IntervalData, get_interval

Real = Union[float, int]
InfoDict = Optional[Dict[str, Any]]
TrackerDataType = Union["TrackerBase", str]


class FinishedSimulation(StopIteration):
    """ exception for signaling that simulation finished successfully """

    pass


class TrackerBase(metaclass=ABCMeta):
    """ base class for implementing trackers """

    _subclasses: Dict[str, "TrackerBase"] = {}  # all inheriting classes

    @fill_in_docstring
    def __init__(self, interval: IntervalData = 1):
        """
        Args:
            interval:
                {ARG_TRACKER_INTERVAL}
        """
        self.interval = get_interval(interval)
        self._logger = logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """ register all subclassess to reconstruct them later """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name"):
            cls._subclasses[cls.name] = cls

    @classmethod
    def from_data(cls, data: TrackerDataType, **kwargs) -> "TrackerBase":
        """create tracker class from given data

        Args:
            data (str or TrackerBase): Data describing the tracker

        Returns:
            :class:`TrackerBase`: An instance representing the tracker
        """
        if isinstance(data, TrackerBase):
            return data
        elif isinstance(data, str):
            try:
                tracker_cls = cls._subclasses[data]
            except KeyError:
                raise ValueError(f"Tracker `{data}` is not defined")
            return tracker_cls(**kwargs)  # type: ignore
        else:
            raise ValueError(f"Unsupported tracker format: `{data}`.")

    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
        """initialize the tracker with information about the simulation

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        if info is not None:
            t_start = info.get("solver", {}).get("t_start", 0)
        else:
            t_start = 0
        return self.interval._initialize(t_start)

    @abstractmethod
    def handle(self, field: FieldBase, t: float) -> None:
        """handle data supplied to this tracker

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """
        pass

    def finalize(self, info: InfoDict = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        pass


TrackerCollectionDataType = Union[List[TrackerDataType], TrackerDataType, None]


class TrackerCollection:
    """List of trackers providing methods to handle them efficiently

    Attributes:
        trackers (list):
            List of the trackers in the collection
    """

    tracker_action_times: List[float]
    """ list: Times at which the trackers need to be handled next """
    time_next_action: float
    """ float: The time of the next interrupt of the simulation """

    def __init__(self, trackers: Optional[List[TrackerBase]] = None):
        """
        Args:
            trackers: List of trackers that are to be handled.
        """
        if trackers is None:
            self.trackers: List[TrackerBase] = []
        elif not hasattr(trackers, "__iter__"):
            raise ValueError(f"`trackers` must be a list of trackers, not {trackers}")
        else:
            self.trackers = trackers

        # do not check trackers before everything was initialized
        self.tracker_action_times = []
        self.time_next_action = np.inf

    @classmethod
    def from_data(
        cls, data: TrackerCollectionDataType, **kwargs
    ) -> "TrackerCollection":
        """create tracker collection from given data

        Args:
            data: Data describing the tracker collection

        Returns:
            :class:`TrackerCollection`:
            An instance representing the tracker collection
        """
        if data is None:
            trackers: List[TrackerBase] = []
        elif isinstance(data, TrackerCollection):
            trackers = data.trackers
        elif isinstance(data, TrackerBase):
            trackers = [data]
        elif isinstance(data, str):
            trackers = [TrackerBase.from_data(data, **kwargs)]
        else:
            trackers = [
                TrackerBase.from_data(tracker)
                for tracker in data
                if tracker is not None
            ]

        return cls(trackers)

    def initialize(self, field: FieldBase, info: InfoDict = None) -> float:
        """initialize the tracker with information about the simulation

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        # initialize trackers and get their action times
        self.tracker_action_times = [
            tracker.initialize(field, info) for tracker in self.trackers
        ]

        if self.trackers:
            # determine next time to check trackers
            self.time_next_action = min(self.tracker_action_times)
        else:
            self.time_next_action = np.inf

        return self.time_next_action

    def handle(self, state: FieldBase, t: float, atol: float = 1.0e-8) -> float:
        """handle all trackers

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
            atol (float):
                An absolute tolerance that is used to determine whether a
                tracker should be called now or whether the simulation should be
                carried on more timesteps. This is basically used to predict the
                next time to decided which one is closer.

        Returns:
            float: The next time the simulation needs to be interrupted to
            handle a tracker.
        """
        # check each tracker to see whether we need to handle it
        stop_iteration_err = None
        for i, t_next in enumerate(self.tracker_action_times):
            if t > t_next or np.isclose(t, t_next, atol=atol, rtol=0):
                try:
                    self.trackers[i].handle(state, t)
                except StopIteration as err:
                    # stop iteration after all trackers have been handled
                    stop_iteration_err = err

                # calculate next event (may skip some if too close)
                self.tracker_action_times[i] = self.trackers[i].interval.next(t)

        if stop_iteration_err is not None:
            raise stop_iteration_err

        # determine next time for checking handler
        if self.trackers:
            self.time_next_action = min(self.tracker_action_times)
        return self.time_next_action

    def finalize(self, info: InfoDict = None) -> None:
        """finalize the tracker, supplying additional information

        Args:
            info (dict):
                Extra information from the simulation
        """
        for tracker in self.trackers:
            tracker.finalize(info=info)
