"""
Defines the :class:`~pde.controller.Controller` class for solving pdes.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


import datetime
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Tuple, TypeVar, Union  # @UnusedImport

from ..tools import numba
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
    """class controlling a simulation"""

    # set a function to determine the current time for profiling purposes. We generally
    # use the more accurate time.process_time, but better performance may be obtained by
    # the faster time.time. This will only affect simulations with many iterations.
    get_current_time = time.process_time

    def __init__(
        self,
        solver: SolverBase,
        t_range: TRangeType,
        tracker: TrackerCollectionDataType = "auto",
    ):
        """
        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                Solver instance that is used to advance the simulation in time
            t_range (float or tuple):
                Sets the time range for which the simulation is run. If only a single
                value `t_end` is given, the time range is assumed to be `[0, t_end]`.
            tracker:
                Defines a tracker that process the state of the simulation at specified
                time intervals. A tracker is either an instance of
                :class:`~pde.trackers.base.TrackerBase` or a string, which identifies a
                tracker. All possible identifiers can be obtained by calling
                :func:`~pde.trackers.base.get_named_trackers`. Multiple trackers can be
                specified as a list. The default value `auto` checks the state for
                consistency (tracker 'consistency') and displays a progress bar (tracker
                'progress') when :mod:`tqdm` is installed. More general trackers are
                defined in :mod:`~pde.trackers`, where all options are explained in
                detail. In particular, the interval at which the tracker is evaluated
                can be chosen when creating a tracker object explicitly.
        """
        self.solver = solver
        self.t_range = t_range  # type: ignore
        self.trackers = TrackerCollection.from_data(tracker)

        self.info: Dict[str, Any] = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def t_range(self) -> Tuple[float, float]:
        """tuple: start and end time of the simulation"""
        return self._t_range

    @t_range.setter
    def t_range(self, value: TRangeType):
        """set start and end time of the simulation

        Args:
            value (float or tuple):
                Set the time range of the simulation. If a single number is given, it
                specifies the final time and the start time is set to zero. If a tuple
                of two numbers is given they are used as start and end time.
        """
        # determine time range
        try:
            self._t_range: Tuple[float, float] = (0, float(value))  # type: ignore
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
        `diagnostics` property of the instance after this function has been called.

        Args:
            state:
                The initial state of the simulation. This state will be copied and thus
                not modified by the simulation. Instead, the final state will be
                returned and trackers can be used to record intermediate states.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default value
                based on the stepper will be chosen.

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
            "package_version": __version__,
            "solver": self.solver.info,
        }

        # initialize trackers
        self.trackers.initialize(state, info=self.diagnostics)

        def _handle_stop_iteration(err):
            """helper function for handling interrupts raised by trackers"""
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
        jit_count_init = numba.JIT_COUNT
        stepper = self.solver.make_stepper(state=state, dt=dt)
        self.diagnostics["jit_count"] = {
            "make_stepper": numba.JIT_COUNT - jit_count_init
        }
        jit_count_after_init = numba.JIT_COUNT

        # initialize profiling information
        solver_start = datetime.datetime.now()
        self.info["solver_start"] = str(solver_start)

        get_time = self.get_current_time  # type: ignore
        profiler = {"solver": 0.0, "tracker": 0.0}
        self.info["profiler"] = profiler
        prof_start_tracker = get_time()

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

                prof_start_solve = get_time()
                profiler["tracker"] += prof_start_solve - prof_start_tracker

                # advance the system to the new time point
                t = stepper(state, t, t_break)

                prof_start_tracker = get_time()
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
        profiler["tracker"] += get_time() - prof_start_tracker
        duration = datetime.datetime.now() - solver_start
        self.info["solver_duration"] = str(duration)
        self.info["t_final"] = t
        jit_count = numba.JIT_COUNT - jit_count_after_init
        self.diagnostics["jit_count"]["simulation"] = jit_count
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
