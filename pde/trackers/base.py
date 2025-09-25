"""Base classes for trackers.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import numpy as np

from ..fields.base import FieldBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import module_available
from .interrupts import InterruptData, parse_interrupt

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for trackers."""

InfoDict = Optional[dict[str, Any]]
TrackerDataType = Union["TrackerBase", str]


class FinishedSimulation(StopIteration):
    """Exception for signaling that simulation finished successfully."""


class TrackerBase(metaclass=ABCMeta):
    """Base class for implementing trackers."""

    _logger: logging.Logger
    _subclasses: dict[str, type[TrackerBase]] = {}  # all inheriting classes

    @fill_in_docstring
    def __init__(self, interrupts: InterruptData = 1):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
        """
        self.interrupt = parse_interrupt(interrupts)

    def __init_subclass__(cls, **kwargs):
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)

        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

        # register all subclasses to reconstruct them later
        if hasattr(cls, "name"):
            assert cls.name != "auto"
            cls._subclasses[cls.name] = cls

    @classmethod
    def from_data(cls, data: TrackerDataType, **kwargs) -> TrackerBase:
        """Create tracker class from given data.

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
            except KeyError as err:
                trackers = sorted(cls._subclasses.keys())
                raise ValueError(f"Tracker `{data}` is not in {trackers}") from err
            return tracker_cls(**kwargs)
        else:
            raise ValueError(f"Unsupported tracker format: `{data}`.")

    def initialize(self, field: FieldBase, info: InfoDict | None = None) -> float:
        """Initialize the tracker with information about the simulation.

        Args:
            field (:class:`~pde.fields.FieldBase`):
                An example of the data that will be analyzed by the tracker
            info (dict):
                Extra information from the simulation

        Returns:
            float: The first time the tracker needs to handle data
        """
        if info is not None:
            t_start = info.get("controller", {}).get("t_start", 0)
        else:
            t_start = 0
        return self.interrupt.initialize(t_start)

    @abstractmethod
    def handle(self, field: FieldBase, t: float) -> None:
        """Handle data supplied to this tracker.

        Args:
            field (:class:`~pde.fields.FieldBase`):
                The current state of the simulation
            t (float):
                The associated time
        """

    def finalize(self, info: InfoDict | None = None) -> None:
        """Finalize the tracker, supplying additional information.

        Args:
            info (dict):
                Extra information from the simulation
        """


TransformationType = Optional[Callable[[FieldBase, float], FieldBase]]


class TransformedTrackerBase(TrackerBase):
    """Tracker that allows modifying incoming data.

    To support the transformations sub-classes need to call
    :code:`self._transform(field, t)` to obtain the transformed field.
    """

    @fill_in_docstring
    def __init__(
        self,
        interrupts: InterruptData = 1,
        *,
        transformation: TransformationType = None,
    ):
        """
        Args:
            interrupts:
                {ARG_TRACKER_INTERRUPT}
            transformation (callable, optional):
                A function that transforms the current state into a new field or field
                collection, which is then used in the tracker. This allows to process
                derived quantities of the field during calculations. The argument needs
                to be a callable function taking 1 or 2 arguments. The first argument
                always is the current field, while the optional second argument is the
                associated time.
        """
        super().__init__(interrupts=interrupts)
        if transformation is not None and not callable(transformation):
            raise TypeError("`transformation` must be callable")
        self.transformation = transformation
        self._emitted_type_warning = False

    def _transform(self, field: FieldBase, t: float) -> FieldBase:
        """Transforms the field according to the defined transformation."""
        if self.transformation is None:
            # no transformation specified -> just return field
            return field

        if self.transformation.__code__.co_argcount == 1:
            # transformation does not take time argument
            transformed_field = self.transformation(field)  # type: ignore

        else:
            # transformation takes field and time arguments
            transformed_field = self.transformation(field, t)

        # check whether transformed data is a proper field
        if not (self._emitted_type_warning or isinstance(transformed_field, FieldBase)):
            warnings.warn("Applied `transformation` did not return a field.")
            self._emitted_type_warning = True
        return transformed_field


TrackerCollectionDataType = Union[Sequence[TrackerDataType], TrackerDataType, None]


class TrackerCollection:
    """List of trackers providing methods to handle them efficiently.

    Attributes:
        trackers (list):
            List of the trackers in the collection
    """

    tracker_action_times: list[float]
    """ list: Times at which the trackers need to be handled next """
    time_next_action: float
    """ float: The time of the next interrupt of the simulation """

    def __init__(self, trackers: list[TrackerBase] | None = None):
        """
        Args:
            trackers: List of trackers that are to be handled.
        """
        if trackers is None:
            self.trackers: list[TrackerBase] = []
        elif not hasattr(trackers, "__iter__"):
            raise ValueError(f"`trackers` must be a list of trackers, not {trackers}")
        else:
            self.trackers = trackers

        # do not check trackers before everything was initialized
        self.tracker_action_times = []
        self.time_next_action = math.inf

    def __len__(self) -> int:
        """Returns the number of trackers in the collection."""
        return len(self.trackers)

    @classmethod
    def from_data(cls, data: TrackerCollectionDataType, **kwargs) -> TrackerCollection:
        """Create tracker collection from given data.

        Args:
            data: Data describing the tracker collection

        Returns:
            :class:`TrackerCollection`:
            An instance representing the tracker collection
        """
        if data == "auto":
            if module_available("tqdm"):
                data = ("progress", "consistency")
            else:
                data = "consistency"

        if data is None:
            trackers: list[TrackerBase] = []
        elif isinstance(data, TrackerCollection):
            trackers = data.trackers
        elif isinstance(data, TrackerBase):
            trackers = [data]
        elif isinstance(data, str):
            trackers = [TrackerBase.from_data(data, **kwargs)]
        elif isinstance(data, (list, tuple)):
            # initialize trackers from a sequence
            trackers, interrupt_ids = [], set()
            for tracker in data:
                if tracker is not None:
                    tracker_obj = TrackerBase.from_data(tracker)
                    if id(tracker_obj.interrupt) in interrupt_ids:
                        # make sure that different trackers never use the same interrupt
                        # class, which would be problematic during iteration
                        tracker_obj.interrupt = tracker_obj.interrupt.copy()
                    interrupt_ids.add(id(tracker_obj.interrupt))
                    trackers.append(tracker_obj)
        else:
            raise TypeError(f"Cannot initialize trackers from class `{data.__class__}`")

        return cls(trackers)

    def initialize(self, field: FieldBase, info: InfoDict | None = None) -> float:
        """Initialize the tracker with information about the simulation.

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
            self.time_next_action = math.inf

        return self.time_next_action

    def handle(self, state: FieldBase, t: float, atol: float = 1.0e-8) -> float:
        """Handle all trackers.

        Args:
            state (:class:`~pde.fields.FieldBase`):
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
            if t > t_next - atol:
                try:
                    self.trackers[i].handle(state, t)
                except StopIteration as err:
                    # This tracker requested to stop the iteration. We save this
                    # information for later, so we can first handle all trackers.
                    stop_iteration_err = err

                # calculate next event (may skip some if too close)
                self.tracker_action_times[i] = self.trackers[i].interrupt.next(t)

        if stop_iteration_err is not None:
            raise stop_iteration_err

        # determine next time for checking handler
        if self.trackers:
            self.time_next_action = min(self.tracker_action_times)
        return self.time_next_action

    def finalize(self, info: InfoDict | None = None) -> None:
        """Finalize the tracker, supplying additional information.

        Args:
            info (dict):
                Extra information from the simulation
        """
        for tracker in self.trackers:
            tracker.finalize(info=info)


def get_named_trackers() -> dict[str, type[TrackerBase]]:
    """Returns all named trackers.

    Returns:
        dict: a mapping of names to the actual tracker classes.
    """
    return TrackerBase._subclasses.copy()
