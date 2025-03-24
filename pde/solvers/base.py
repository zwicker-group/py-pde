"""Package that contains base classes for solvers.

Beside the abstract base class defining the interfaces, we also provide
:class:`AdaptiveSolverBase`, which contains methods for implementing adaptive solvers.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import warnings
from abc import ABCMeta
from inspect import isabstract
from typing import Any, Callable

import numba as nb
import numpy as np
from numba.extending import register_jitable

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from ..tools.math import OnlineStatistics
from ..tools.misc import classproperty
from ..tools.numba import is_jitted, jit
from ..tools.typing import BackendType, StepperHook

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for solvers."""


class ConvergenceError(RuntimeError):
    """Indicates that an implicit step did not converge."""


class SolverBase:
    """Base class for PDE solvers."""

    dt_default: float = 1e-3
    """float: default time step used if no time step was specified"""

    _use_post_step_hook: bool = True
    """bool: flag choosing whether the post-step hook of the PDE is called"""

    _mpi_synchronization: bool = False
    """bool: Flag indicating whether MPI synchronization is required. This is never the
    case for serial solvers and even parallelized solvers might set this flag to False
    if no synchronization between nodes is required"""

    _subclasses: dict[str, type[SolverBase]] = {}
    """dict: dictionary of all inheriting classes"""

    _logger: logging.Logger

    def __init__(self, pde: PDEBase, *, backend: BackendType = "auto"):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.
        """
        self.pde = pde
        self.backend = backend
        self.info: dict[str, Any] = {"class": self.__class__.__name__}
        if self.pde:
            self.info["pde_class"] = self.pde.__class__.__name__

    def __init_subclass__(cls, **kwargs):
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)

        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

        # register all subclasses to reconstruct them later
        if not isabstract(cls):
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}")
            cls._subclasses[cls.__name__] = cls
        if hasattr(cls, "name") and cls.name:
            if cls.name in cls._subclasses:
                _base_logger.warning("Solver `%s` is already registered", cls.name)
            cls._subclasses[cls.name] = cls

    @classmethod
    def from_name(cls, name: str, pde: PDEBase, **kwargs) -> SolverBase:
        r"""Create solver class based on its name.

        Solver classes are automatically registered when they inherit from
        :class:`SolverBase`. Note that this also requires that the respective python
        module containing the solver has been loaded before it is attempted to be used.

        Args:
            name (str):
                The name of the solver to construct
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            \**kwargs:
                Additional arguments for the constructor of the solver

        Returns:
            An instance of a subclass of :class:`SolverBase`
        """
        try:
            # obtain the solver class associated with `name`
            solver_class = cls._subclasses[name]
        except KeyError:
            # solver was not registered
            solvers = (
                f"'{solver}'"
                for solver in sorted(cls._subclasses.keys())
                if not solver.endswith("Solver")
            )
            raise ValueError(
                f"Unknown solver method '{name}'. Registered solvers are "
                + ", ".join(solvers)
            ) from None

        return solver_class(pde, **kwargs)

    @classproperty
    def registered_solvers(cls) -> list[str]:
        """list of str: the names of the registered solvers"""
        return sorted(cls._subclasses.keys())

    @property
    def _compiled(self) -> bool:
        """bool: indicates whether functions need to be compiled"""
        return self.backend == "numba" and not nb.config.DISABLE_JIT

    def _make_error_synchronizer(
        self, operator: int | str = "MAX"
    ) -> Callable[[float], float]:
        """Return function that synchronizes errors between multiple processes.

        Args:
            operator (str or int):
                Flag determining how the value from multiple nodes is combined.
                Possible values include "MAX", "MIN", and "SUM".

        Returns:
            Function that can be used to synchronize errors across nodes
        """
        if self._mpi_synchronization:  # mpi.parallel_run:
            # in a parallel run, we need to synchronize values
            from ..tools.mpi import mpi_allreduce

            @register_jitable
            def synchronize_errors(error: float) -> float:
                """Return error synchronized accross all cores."""
                return mpi_allreduce(error, operator=operator)  # type: ignore

        else:

            @register_jitable
            def synchronize_errors(value: float) -> float:
                return value

        return synchronize_errors  # type: ignore

    def _make_post_step_hook(self, state: FieldBase) -> StepperHook:
        """Create a function that calls the post-step hook of the PDE.

        A no-op function is returned if :attr:`SolverBase._use_post_step_hook` is
        `False` or the PDE does not provide :meth:`PDEBase.make_post_step_hook`.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.

        Returns:
            callable: The jit-able function that calls the post-step hook
        """
        post_step_hook: StepperHook | None = None

        if self._use_post_step_hook:
            try:
                # look for the definition of a hook function
                if hasattr(self.pde, "make_modify_after_step"):
                    # Deprecated on 2024-08-02
                    warnings.warn(
                        "`make_modify_after_step` has been replaced by `make_post_step_hook`",
                        DeprecationWarning,
                    )
                    modify_after_step = self.pde.make_modify_after_step(state)
                    if self._compiled:
                        sig_modify = (nb.typeof(state.data),)
                        modify_after_step = jit(sig_modify)(modify_after_step)

                    def post_step_hook(
                        state_data: np.ndarray, t: float, post_step_data: np.ndarray
                    ):
                        """Wrap function to adjust signature."""
                        post_step_data += modify_after_step(state_data)

                    # create zero of correct type
                    self._post_step_data_init = np.dtype(state.dtype).type(0)
                    self._logger.info(
                        "Created post-step hook from `make_modify_after_step`"
                    )

                else:
                    # get hook function and initial data from PDE
                    post_step_hook, self._post_step_data_init = (
                        self.pde.make_post_step_hook(state)
                    )
                    self._logger.info("Created post-step hook from PDE")

            except NotImplementedError:
                pass  # no hook function defined on the PDE

        if post_step_hook is None:
            # hook function is not necessary or was not supplied -> provide no-op

            def post_step_hook(
                state_data: np.ndarray, t: float, post_step_data: np.ndarray
            ):
                """Default hook function does nothing."""

            self._post_step_data_init = None
            self._logger.debug("No post-step hook defined")

        else:
            # ensure that the initial values is a mutable array
            self._post_step_data_init = np.array(self._post_step_data_init, copy=True)

        self._post_step_data_type = nb.typeof(self._post_step_data_init)
        if self._compiled:
            sig_hook = (nb.typeof(state.data), nb.float64, self._post_step_data_type)
            post_step_hook = jit(sig_hook)(post_step_hook)
            self._logger.debug("Compiled post-step hook")

        return post_step_hook  # type: ignore

    def _make_pde_rhs(
        self, state: FieldBase, backend: BackendType = "auto"
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """Obtain a function for evaluating the right hand side.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.

        Raises:
            RuntimeError: when a stochastic partial differential equation is encountered
            but `allow_stochastic == False`.

        Returns:
            A function that is called with data given by a :class:`~numpy.ndarray` and a
            time. The function returns the deterministic evolution rate and (if
            applicable) a realization of the associated noise.
        """
        if getattr(self.pde, "is_sde", False):
            raise RuntimeError(
                "Cannot create a deterministic stepper for a stochastic equation"
            )

        rhs = self.pde.make_pde_rhs(state, backend=backend)  # type: ignore

        if hasattr(rhs, "_backend"):
            self.info["backend"] = rhs._backend
        elif is_jitted(rhs):
            self.info["backend"] = "numba"
        else:
            self.info["backend"] = "undetermined"

        return rhs

    def _make_sde_rhs(
        self, state: FieldBase, backend: str = "auto"
    ) -> Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]:
        """Obtain a function for evaluating the right hand side.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.

        Raises:
            RuntimeError: when a stochastic partial differential equation is encountered
            but `allow_stochastic == False`.

        Returns:
            A function that is called with data given by a :class:`~numpy.ndarray` and a
            time. The function returns the deterministic evolution rate and (if
            applicable) a realization of the associated noise.
        """
        rhs = self.pde.make_sde_rhs(state, backend=backend)  # type: ignore

        if hasattr(rhs, "_backend"):
            self.info["backend"] = rhs._backend
        elif is_jitted(rhs):
            self.info["backend"] = "numba"
        else:
            self.info["backend"] = "undetermined"

        return rhs

    def _make_single_step_fixed_dt(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float], None]:
        """Return a function doing a single step with a fixed time step.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.
        """
        raise NotImplementedError("Fixed stepper has not been defined")

    def _make_fixed_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, int, Any], float]:
        """Return a stepper function using an explicit scheme with fixed time steps.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.
        """
        single_step = self._make_single_step_fixed_dt(state, dt)
        post_step_hook = self._make_post_step_hook(state)

        if self._compiled:
            sig_single_step = (nb.typeof(state.data), nb.double)
            single_step = jit(sig_single_step)(single_step)

        def fixed_stepper(
            state_data: np.ndarray, t_start: float, steps: int, post_step_data
        ) -> float:
            """Perform `steps` steps with fixed time steps."""
            for i in range(steps):
                # calculate the right hand side
                t = t_start + i * dt
                single_step(state_data, t)
                post_step_hook(state_data, t, post_step_data)

            return t + dt

        if self._compiled:
            sig_fixed = (
                nb.typeof(state.data),
                nb.double,
                nb.int_,
                self._post_step_data_type,
            )
            fixed_stepper = jit(sig_fixed)(fixed_stepper)

        return fixed_stepper

    def make_stepper(
        self, state: FieldBase, dt: float | None = None
    ) -> Callable[[FieldBase, float, float], float]:
        """Return a stepper function using an explicit scheme.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step used (Uses :attr:`SolverBase.dt_default` if `None`)

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        # support `None` as a default value, so the controller can signal that
        # the solver should use a default time step
        if dt is None:
            dt = self.dt_default
            self._logger.warning(
                "Explicit stepper with a fixed time step did not receive any "
                "initial value for `dt`. Using dt=%g, but specifying a value or "
                "enabling adaptive stepping is advisable.",
                dt,
            )
        dt_float = float(dt)  # explicit casting to help type checking

        # create stepper with fixed steps
        fixed_stepper = self._make_fixed_stepper(state, dt_float)

        self.info["dt"] = dt_float
        self.info["steps"] = 0
        self.info["post_step_data"] = self._post_step_data_init
        self.info["stochastic"] = getattr(self.pde, "is_sde", False)
        # We don't access self.pde directly since we might want to reuse the solver
        # infrastructure for more general cases where a PDE is not defined.

        def wrapped_stepper(state: FieldBase, t_start: float, t_end: float) -> float:
            """Advance `state` from `t_start` to `t_end` using fixed steps."""
            # retrieve last post_step_data and continue with this
            post_step_data = self.info["post_step_data"]
            # calculate number of steps that lead to an end time closest to t_end
            steps = max(1, int(round((t_end - t_start) / dt_float)))
            # call the stepper with fixed time steps
            t_last = fixed_stepper(state.data, t_start, steps, post_step_data)
            # keep some stats and data
            self.info["steps"] += steps
            return t_last

        return wrapped_stepper


class AdaptiveSolverBase(SolverBase):
    """Base class for adaptive time steppers."""

    dt_min: float = 1e-10
    """float: minimal time step that the adaptive solver will use"""
    dt_max: float = 1e10
    """float: maximal time step that the adaptive solver will use"""

    def __init__(
        self,
        pde: PDEBase,
        *,
        backend: BackendType = "auto",
        adaptive: bool = True,
        tolerance: float = 1e-4,
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.
            adaptive (bool):
                When enabled, the time step is adjusted during the simulation using the
                error tolerance set with `tolerance`.
            tolerance (float):
                The error tolerance used in adaptive time stepping. This is used in
                adaptive time stepping to choose a time step which is small enough so
                the truncation error of a single step is below `tolerance`.
        """
        super().__init__(pde, backend=backend)
        self.adaptive = adaptive
        self.tolerance = tolerance

    def _make_dt_adjuster(self) -> Callable[[float, float], float]:
        """Return a function that can be used to adjust time steps."""
        dt_min = self.dt_min
        dt_min_nan_err = f"Encountered NaN even though dt < {dt_min}"
        dt_min_err = f"Time step below {dt_min}"
        dt_max = self.dt_max

        def adjust_dt(dt: float, error_rel: float) -> float:
            """Helper function that adjust the time step.

            The goal is to keep the relative error `error_rel` close to 1.

            Args:
                dt (float): Current time step
                error_rel (float): Current (normalized) error estimate

            Returns:
                float: Time step of the next iteration
            """
            # adjust the time step
            if error_rel < 0.00057665:
                # error was very small => maximal increase in dt
                # The constant on the right hand side of the comparison is chosen to
                # agree with the equation for adjusting dt below
                dt *= 4.0
            elif np.isnan(error_rel):
                # state contained NaN => decrease time step strongly
                dt *= 0.25
            else:
                # otherwise, adjust time step according to error
                dt *= max(0.9 * error_rel**-0.2, 0.1)

            # limit time step to permissible bracket
            if dt > dt_max:
                dt = dt_max
            elif dt < dt_min:
                if np.isnan(error_rel):
                    raise RuntimeError(dt_min_nan_err)
                else:
                    raise RuntimeError(dt_min_err)

            return dt

        if self._compiled:
            adjust_dt = jit((nb.double, nb.double))(adjust_dt)

        return adjust_dt

    def _make_single_step_variable_dt(
        self, state: FieldBase
    ) -> Callable[[np.ndarray, float, float], np.ndarray]:
        """Return a function doing a single step with a variable time step.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is
            `(state: numpy.ndarray, t_start: float, t_end: float)`
        """
        rhs_pde = self._make_pde_rhs(state, backend=self.backend)

        def single_step(state_data: np.ndarray, t: float, dt: float) -> np.ndarray:
            """Basic implementation of Euler scheme."""
            return state_data + dt * rhs_pde(state_data, t)  # type: ignore

        return single_step

    def _make_single_step_error_estimate(
        self, state: FieldBase
    ) -> Callable[[np.ndarray, float, float], tuple[np.ndarray, float]]:
        """Make a stepper that also estimates the error.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
        """
        if getattr(self.pde, "is_sde", False):
            raise RuntimeError("Cannot use adaptive stepper with stochastic equation")

        single_step = self._make_single_step_variable_dt(state)
        if self._compiled:
            sig_single_step = (nb.typeof(state.data), nb.double, nb.double)
            single_step = jit(sig_single_step)(single_step)

        def single_step_error_estimate(
            state_data: np.ndarray, t: float, dt: float
        ) -> tuple[np.ndarray, float]:
            """Basic stepper to estimate error."""
            # single step with dt
            k1 = single_step(state_data, t, dt)

            # double step with half the time step
            k2a = single_step(state_data, t, 0.5 * dt)
            k2 = single_step(k2a, t + 0.5 * dt, 0.5 * dt)

            # calculate maximal error
            error = np.abs(k1 - k2).max()

            return k2, error

        return single_step_error_estimate

    def _make_adaptive_stepper(
        self, state: FieldBase
    ) -> Callable[
        [np.ndarray, float, float, float, OnlineStatistics | None, Any],
        tuple[float, float, int],
    ]:
        """Make an adaptive Euler stepper.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        # obtain functions determining how the PDE is evolved
        single_step_error = self._make_single_step_error_estimate(state)
        post_step_hook = self._make_post_step_hook(state)
        sync_errors = self._make_error_synchronizer()

        # obtain auxiliary functions
        adjust_dt = self._make_dt_adjuster()
        tolerance = self.tolerance
        dt_min = self.dt_min

        if self._compiled:
            # compile paired stepper
            sig_stepper = (nb.typeof(state.data), nb.double, nb.double)
            single_step_error = jit(sig_stepper)(single_step_error)

        def adaptive_stepper(
            state_data: np.ndarray,
            t_start: float,
            t_end: float,
            dt_init: float,
            dt_stats: OnlineStatistics | None = None,
            post_step_data=None,
        ) -> tuple[float, float, int]:
            """Adaptive stepper that advances the state in time."""
            dt_opt = dt_init
            t = t_start
            steps = 0
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = max(min(dt_opt, t_end - t), dt_min)

                # two different steppings to estimate errors
                new_state, error = single_step_error(state_data, t, dt_step)

                error_rel = error / tolerance  # normalize error to given tolerance
                # synchronize the error between all processes (necessary for MPI)
                error_rel = sync_errors(error_rel)

                # do the step if the error is sufficiently small
                if error_rel <= 1:
                    steps += 1
                    t += dt_step
                    state_data[...] = new_state
                    post_step_hook(state_data, t, post_step_data)

                    if dt_stats is not None:
                        dt_stats.add(dt_step)

                if t < t_end:
                    # adjust the time step and continue (happens in every MPI process)
                    dt_opt = adjust_dt(dt_step, error_rel)
                else:
                    break  # return to the controller

            return t, dt_opt, steps

        if self._compiled:
            # compile inner stepper
            sig_adaptive = (
                nb.typeof(state.data),
                nb.double,
                nb.double,
                nb.double,
                nb.typeof(self.info["dt_statistics"]),
                self._post_step_data_type,
            )
            adaptive_stepper = jit(sig_adaptive)(adaptive_stepper)

        self._logger.info("Initialized adaptive stepper")
        return adaptive_stepper

    def make_stepper(
        self, state: FieldBase, dt: float | None = None
    ) -> Callable[[FieldBase, float, float], float]:
        """Return a stepper function using an explicit scheme.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step used (Uses :attr:`SolverBase.dt_default` if `None`). This sets
                the initial time step for adaptive solvers.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        if not self.adaptive:
            # create stepper with fixed steps
            return super().make_stepper(state, dt)

        if getattr(self.pde, "is_sde", False):
            # adaptive steppers cannot deal with stochastic PDEs
            raise RuntimeError("Cannot use adaptive stepper with stochastic equation")

        # Support `None` as a default value, so the controller can signal that
        # the solver should use a default time step.
        if dt is None:
            dt_float = self.dt_default
        else:
            dt_float = float(dt)  # explicit casting to help type checking

        # create stepper with adaptive steps
        self.info["dt_statistics"] = OnlineStatistics()
        adaptive_stepper = self._make_adaptive_stepper(state)

        self.info["dt"] = dt_float
        self.info["dt_adaptive"] = True
        self.info["steps"] = 0
        self.info["stochastic"] = getattr(self.pde, "is_sde", False)
        self.info["post_step_data"] = self._post_step_data_init

        def wrapped_stepper(state: FieldBase, t_start: float, t_end: float) -> float:
            """Advance `state` from `t_start` to `t_end` using adaptive steps."""
            nonlocal dt_float  # `dt_float` stores value for the next call
            # retrieve last post_step_data and continue with this
            post_step_data = self.info["post_step_data"]
            # call the adaptive stepper
            t_last, dt_float, steps = adaptive_stepper(
                state.data,
                t_start,
                t_end,
                dt_float,
                self.info["dt_statistics"],
                post_step_data,
            )
            # keep some stats and data
            self.info["steps"] += steps
            return t_last

        return wrapped_stepper
