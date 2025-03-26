"""Defines a class controlling the simulations of PDEs.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union

from .. import __version__
from ..tools.numba import JIT_COUNT
from ..trackers.base import (
    FinishedSimulation,
    TrackerCollection,
    TrackerCollectionDataType,
)
from .base import SolverBase

if TYPE_CHECKING:
    from ..fields.base import FieldBase

_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger for controller."""

TRangeType = Union[float, tuple[float, float]]
TState = TypeVar("TState", bound="FieldBase")


class Controller:
    """Class controlling a simulation.

    The controller calls a solver to advance the simulation into the future and it takes
    care of trackers that analyze and modify the state periodically. The controller also
    handles errors in the simulations and the trackers, as well as user-induced
    interrupts, e.g., by hitting Ctrl-C or Cmd-C to cause a :class:`KeyboardInterrupt`.
    In case of problems, the Controller writes additional information into
    :attr:`~Controller.diagnostics`, which can help to diagnose problems.
    """

    diagnostics: dict[str, Any]
    """dict: diagnostic information (available after simulation finished)"""

    _get_current_time: Callable = time.process_time
    """callable: function to determine the current time for profiling purposes. We
    generally use the more accurate :func:`time.process_time`, but better performance
    may be obtained by the faster :func:`time.time`. This will only affect simulations
    with many iterations."""

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
                Defines trackers that process the state of the simulation at specified
                times. A tracker is either an instance of
                :class:`~pde.trackers.base.TrackerBase` or a string identifying a
                tracker (possible identifiers can be obtained by calling
                :func:`~pde.trackers.base.get_named_trackers`). Multiple trackers can be
                specified as a list. The default value `auto` checks the state for
                consistency (tracker 'consistency') and displays a progress bar (tracker
                'progress') when :mod:`tqdm` is installed. More general trackers are
                defined in :mod:`~pde.trackers`, where all options are explained in
                detail. In particular, the time points where the tracker analyzes data
                can be chosen when creating a tracker object explicitly.
        """
        self.solver = solver
        self.t_range = t_range  # type: ignore
        self.trackers = TrackerCollection.from_data(tracker)

        # initialize some diagnostic information
        self.info: dict[str, Any] = {}
        self.diagnostics = {
            "controller": self.info,
            "package_version": __version__,
        }

    @property
    def t_range(self) -> tuple[float, float]:
        """tuple: start and end time of the simulation"""
        return self._t_range

    @t_range.setter
    def t_range(self, value: TRangeType):
        """Set start and end time of the simulation.

        Args:
            value (float or tuple):
                Set the time range of the simulation. If a single number is given, it
                specifies the final time and the start time is set to zero. If a tuple
                of two numbers is given they are used as start and end time.
        """
        # determine time range
        try:
            self._t_range: tuple[float, float] = (0, float(value))  # type: ignore
        except TypeError as err:  # assume a single number was given
            if len(value) == 2:  # type: ignore
                self._t_range = tuple(value)  # type: ignore
            else:
                raise ValueError(
                    "t_range must be set to a single number or a tuple of two numbers"
                ) from err

    def _get_stop_handler(self) -> Callable[[Exception, float], tuple[int, str]]:
        """Return function that handles messaging."""

        def _handle_stop_iteration(err: Exception, t: float) -> tuple[int, str]:
            """Helper function for handling interrupts raised by trackers."""
            if isinstance(err, FinishedSimulation):
                # tracker determined that the simulation finished
                self.info["successful"] = True
                msg = f"Simulation finished at t={t}"
                msg_level = logging.INFO
                if hasattr(err, "value") and err.value:
                    self.info["stop_reason"] = err.value
                    msg += f" ({err.value})"
                else:
                    self.info["stop_reason"] = "Tracker raised FinishedSimulation"

            else:
                # tracker determined that there was a problem
                self.info["successful"] = False
                msg = f"Simulation aborted at t={t}"
                msg_level = logging.WARNING
                if hasattr(err, "value") and err.value:
                    self.info["stop_reason"] = err.value
                    msg += f" ({err.value})"
                else:
                    self.info["stop_reason"] = "Tracker raised StopIteration"

            return msg_level, msg

        return _handle_stop_iteration

    def _run_main_process(self, state: TState, dt: float | None = None) -> None:
        """Run the main part of the simulation.

        This is either a serial run or the main node of an MPI run. Diagnostic
        information about the solver procedure are available in the `diagnostics`
        property of the instance after this function has been called.

        Args:
            state:
                The initial state, which will be updated during the simulation.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default value
                based on the stepper will be chosen.
        """
        # gather basic information
        t_start, t_end = self.t_range
        get_time = self._get_current_time

        # initialize solver information
        self.info["t_start"] = t_start
        self.info["t_end"] = t_end
        self.diagnostics["solver"] = getattr(self.solver, "info", {})

        # initialize profilers
        jit_count_base = int(JIT_COUNT)
        profiler = {"solver": 0.0, "tracker": 0.0}
        self.info["profiler"] = profiler
        prof_start_compile = get_time()

        # initialize trackers and handlers
        self.trackers.initialize(state, info=self.diagnostics)
        handle_stop_iteration = self._get_stop_handler()

        # initialize the stepper
        stepper = self.solver.make_stepper(state=state, dt=dt)

        # store intermediate profiling information before starting simulation
        jit_count_after_init = int(JIT_COUNT)
        self.info["jit_count"] = {"make_stepper": jit_count_after_init - jit_count_base}
        prof_start_tracker = get_time()
        profiler["compilation"] = prof_start_tracker - prof_start_compile
        solver_start = datetime.datetime.now()
        self.info["solver_start"] = str(solver_start)

        if dt is None:
            # use self.solver.info['dt'] if it is present
            dt = self.diagnostics["solver"].get("dt")
        # add some tolerance to account for inaccurate float point math
        if dt is None:  # self.solver.info['dt'] might be None
            atol = 1e-12
        else:
            atol = 1e-9 * dt

        # evolve the system from t_start to t_end
        t = t_start
        _logger.debug("Start simulation at t=%g", t)
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
            self.info["stop_reason"] = "Tracker raised StopIteration"
            msg_level, msg = handle_stop_iteration(err, t)
            self.diagnostics["last_tracker_time"] = t
            self.diagnostics["last_state"] = state

        except KeyboardInterrupt:
            # iteration has been interrupted by the user
            self.info["successful"] = False
            self.info["stop_reason"] = "User interrupted simulation"
            msg = f"Simulation interrupted at t={t}"
            msg_level = logging.INFO
            self.diagnostics["last_tracker_time"] = t
            self.diagnostics["last_state"] = state

        except Exception:
            # any other exception
            self.diagnostics["last_tracker_time"] = t
            self.diagnostics["last_state"] = state
            raise

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
                msg_level, msg = handle_stop_iteration(err, t)

        # calculate final statistics
        profiler["tracker"] += get_time() - prof_start_tracker
        duration = datetime.datetime.now() - solver_start
        self.info["solver_duration"] = str(duration)
        self.info["t_final"] = t
        self.info["jit_count"]["simulation"] = int(JIT_COUNT) - jit_count_after_init
        self.trackers.finalize(info=self.diagnostics)
        if "dt_statistics" in getattr(self.solver, "info", {}):
            dt_statistics = dict(self.solver.info["dt_statistics"].to_dict())
            self.solver.info["dt_statistics"] = dt_statistics

        # show information after a potential progress bar has been deleted to not mess
        # up the display
        _logger.log(msg_level, msg)
        if profiler["tracker"] > max(profiler["solver"], 1):
            _logger.warning(
                "Spent more time on handling trackers (%.3g) than on the actual "
                "simulation (%.3g)",
                profiler["tracker"],
                profiler["solver"],
            )

    def _run_client_process(self, state: TState, dt: float | None = None) -> None:
        """Run the simulation on client nodes during an MPI run.

        This function just loops the stepper advancing the sub field of the current node
        in time. All other logic, including trackers, are done in the main node.

        Args:
            state:
                The initial state, which will be updated during the simulation.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default value
                based on the stepper will be chosen.

        Returns:
            The state at the final time point.
        """
        # get stepper function
        stepper = self.solver.make_stepper(state=state, dt=dt)

        if not self.solver.info.get("use_mpi", False):
            _logger.warning(
                "Started multiprocessing run without a stepper that supports it. Use "
                "`ExplicitMPISolver` to profit from multiple cores"
            )

        # evolve the system from t_start to t_end
        t_start, t_end = self.t_range
        t = t_start
        while t < t_end:
            t = stepper(state, t, t_end)

    def _run_serial(self, state: TState, dt: float | None = None) -> TState:
        """Run the simulation in serial mode.

        Diagnostic information about the solver are available in the
        :attr:`~Controller.diagnostics` property after this function has been called.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                The initial state of the simulation.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default value
                based on the stepper will be chosen.

        Returns:
            The state at the final time point. If multiprocessing is used, only the main
            node will return the state. All other nodes return None.
        """
        self.info["mpi_run"] = False
        self._run_main_process(state, dt)
        return state

    def _run_parallel(self, state: TState, dt: float | None = None) -> TState | None:
        """Run the simulation in MPI mode.

        Diagnostic information about the solver are available in the
        :attr:`~Controller.diagnostics` property after this function has been called.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                The initial state of the simulation.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default value
                based on the stepper will be chosen.

        Returns:
            The state at the final time point. If multiprocessing is used, only the main
            node will return the state. All other nodes return None.
        """
        from mpi4py import MPI

        from ..tools import mpi

        self.info["mpi_run"] = True
        self.info["mpi_count"] = mpi.size
        self.info["mpi_rank"] = mpi.rank

        if mpi.is_main:
            # this node is the primary one and must thus run the main process
            try:
                self._run_main_process(state, dt)
            except Exception as err:
                print(err)  # simply print the exception to show some info
                _logger.error("Error in main node", exc_info=err)
                time.sleep(0.5)  # give some time for info to propagate
                MPI.COMM_WORLD.Abort()  # abort all other nodes
                raise
            else:
                return state

        else:
            # this node is a secondary node and must thus run the client process
            try:
                self._run_client_process(state, dt)
            except Exception as err:
                print(err)  # simply print the exception to show some info
                _logger.error("Error in node %d", mpi.rank, exc_info=err)
                time.sleep(0.5)  # give some time for info to propagate
                MPI.COMM_WORLD.Abort()  # abort all other (and main) nodes
                raise
            else:
                return None  # do not return anything in client processes

    def run(self, initial_state: TState, dt: float | None = None) -> TState | None:
        """Run the simulation.

        Diagnostic information about the solver are available in the
        :attr:`~Controller.diagnostics` property after this function has been called.

        Args:
            initial_state (:class:`~pde.fields.base.FieldBase`):
                The initial state of the simulation. This state will be copied and thus
                not modified by the simulation. Instead, the final state will be
                returned and trackers can be used to record intermediate states.
            dt (float):
                Time step of the chosen stepping scheme. If `None`, a default value
                based on the stepper will be chosen.

        Returns:
            The state at the final time point. If multiprocessing is used, only the main
            node will return the state. All other nodes return None.
        """
        from ..tools import mpi

        # copy the initial state to not modify the supplied one
        if getattr(self.solver, "pde", None) and self.solver.pde.complex_valued:
            _logger.info("Convert state to complex numbers")
            state: TState = initial_state.copy(dtype=complex)
        else:
            state = initial_state.copy()

        if mpi.size > 1:  # run the simulation on multiple nodes
            return self._run_parallel(state, dt)
        else:
            return self._run_serial(state, dt)
