"""
Defines the :class:`~pde.controller.Controller` class for solving pdes.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import datetime
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Tuple, TypeVar, Union  # @UnusedImport

from ..trackers.base import (
    FinishedSimulation,
    TrackerCollection,
    TrackerCollectionDataType,
)
from ..version import __version__
from .base import SolverBase

if TYPE_CHECKING:
    from ..fields.base import FieldBase  # @UnusedImport


TRangeType = Union[float, Tuple[float, float]]
TState = TypeVar("TState", bound="FieldBase")


class Controller:
    """ class controlling a simulation """

    _t_range: Tuple[float, float]

    def __init__(
        self,
        solver: SolverBase,
        t_range: TRangeType,
        tracker: TrackerCollectionDataType = ["progress", "consistency"],
    ):
        """
        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                Solver instance that is used to advance the simulation in time
            t_range (float or tuple):
                Sets the time range for which the simulation is run. If only a
                single value `t_end` is given, the time range is assumed to be
                `[0, t_end]`.
            tracker:
                Defines trackers that process the state of the simulation at
                fixed time intervals. Multiple trackers can be specified as a
                list. The default value 'auto' is converted to
                `['progress', 'consistency']` for normal simulations. This thus
                displays a progress bar and checks the state for consistency,
                aborting the simulation when not-a-number values appear. To
                disable trackers, set the value to `None`.
        """
        self.solver = solver
        self.t_range = t_range  # type: ignore
        self.trackers = TrackerCollection.from_data(tracker)

        self.info: Dict[str, Any] = {"package_version": __version__}
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def t_range(self) -> Tuple[float, float]:
        return self._t_range

    @t_range.setter
    def t_range(self, value: TRangeType):
        # determine time range
        try:
            self._t_range = 0, float(value)  # type: ignore
        except TypeError:  # assume a single number was given
            if len(value) == 2:  # type: ignore
                self._t_range = tuple(value)  # type: ignore
            else:
                raise ValueError(
                    "t_range must be set to a single number or a tuple of two numbers"
                )

    def run(self, state: TState, dt: float = None) -> TState:
        """run the simulation

        Diagnostic information about the solver procedure are available in the
        `diagnostics` property of the instance after this function has been
        called.

        Args:
            state:
                The initial state of the simulation. This state will be copied
                and thus not modified by the simulation. Instead, the final
                state will be returned and trackers can be used to record
                intermediate states.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default
                value based on the stepper will be chosen.

        Returns:
            The state at the final time point.
        """
        # copy the initial state to not modify the supplied one
        if hasattr(self.solver, "pde") and self.solver.pde.complex_valued:
            self._logger.info("Convert state to complex numbers")
            state = state.copy(dtype="complex")
        else:
            state = state.copy()
        t_start, t_end = self.t_range

        # initialize solver information
        self.info["t_start"] = t_start
        self.info["t_end"] = t_end
        self.info["solver_class"] = self.solver.__class__.__name__
        self.diagnostics: Dict[str, Any] = {
            "controller": self.info,
            "solver": self.solver.info,
        }

        # initialize trackers
        self.trackers.initialize(state, info=self.diagnostics)

        def _handle_stop_iteration(err):
            """ helper function for handling interrupts raised by trackers """
            if isinstance(err, FinishedSimulation):
                # tracker determined that the simulation finished
                self.info["successful"] = True
                msg = f"Simulation finished at t={t}"
                msg_level = logging.INFO
                if err.value:
                    self.info["stop_reason"] = err.value
                    msg += f" ({err.value})"
                else:
                    self.info["stop_reason"] = "Tracker raised FinishedSimulation"

            else:
                # tracker determined that there was a problem
                self.info["successful"] = False
                msg = f"Simulation aborted at t={t}"
                msg_level = logging.WARNING
                if err.value:
                    self.info["stop_reason"] = err.value
                    msg += f" ({err.value})"
                else:
                    self.info["stop_reason"] = "Tracker raised StopIteration"

            return msg_level, msg

        # initialize the stepper
        stepper = self.solver.make_stepper(state=state, dt=dt)

        # initialize profiling information
        solver_start = datetime.datetime.now()
        self.info["solver_start"] = str(solver_start)
        profiler = {"solver": 0.0, "tracker": 0.0}
        self.info["profiler"] = profiler
        prof_start_tracker = time.process_time()

        # add some tolerance to account for inaccurate float point math
        if dt is None:
            dt = self.solver.info.get("dt")
            # Note that self.solver.info['dt'] might be None

        if dt is None:
            atol = 1e-12
        else:
            atol = 0.1 * dt

        # evolve the system from t_start to t_end
        t = t_start
        self._logger.debug(f"Start simulation at t={t}")
        try:
            while t < t_end:
                # determine next time point with an action
                t_next_action = self.trackers.handle(state, t, atol=atol)
                t_next_action = max(t_next_action, t + atol)
                t_break = min(t_next_action, t_end)

                prof_start_solve = time.process_time()
                profiler["tracker"] += prof_start_solve - prof_start_tracker

                # advance the system to the new time point
                t = stepper(state, t, t_break)

                prof_start_tracker = time.process_time()
                profiler["solver"] += prof_start_tracker - prof_start_solve

        except StopIteration as err:
            # iteration has been interrupted by a tracker
            msg_level, msg = _handle_stop_iteration(err)

        except KeyboardInterrupt:
            # iteration has been interrupted by the user
            self.info["successful"] = False
            self.info["stop_reason"] = "User interrupted simulation"
            msg = f"Simulation interrupted at t={t}"
            msg_level = logging.INFO

        else:
            # reached final time
            self.info["successful"] = True
            self.info["stop_reason"] = "Reached final time"
            msg = f"Simulation finished at t={t_end}."
            msg_level = logging.INFO

            # handle trackers one more time when t_end is reached
            try:
                self.trackers.handle(state, t, atol=atol)
            except StopIteration as err:
                # error detected in the final handling of the tracker
                msg_level, msg = _handle_stop_iteration(err)

        # calculate final statistics
        profiler["tracker"] += time.process_time() - prof_start_tracker
        duration = datetime.datetime.now() - solver_start
        self.info["solver_duration"] = str(duration)
        self.info["t_final"] = t
        self.trackers.finalize(info=self.diagnostics)

        # show information after a potential progress bar has been deleted to
        # not mess up the display
        self._logger.log(msg_level, msg)
        if profiler["tracker"] > max(profiler["solver"], 1):
            self._logger.warning(
                f"Spent more time on handling trackers ({profiler['tracker']}) than on "
                f"the actual simulation ({profiler['solver']})"
            )

        return state
