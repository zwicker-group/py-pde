"""Package that contains base classes for solvers.

Beside the abstract base class defining the interfaces, we also provide
:class:`AdaptiveSolverBase`, which contains methods for implementing adaptive solvers.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from inspect import isabstract
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..tools.math import OnlineStatistics
from ..tools.misc import classproperty
from ..tools.typing import BackendType, NumericArray, StepperHook, TField

if TYPE_CHECKING:
    from ..backends.base import BackendBase
    from ..pdes.base import PDEBase


_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for solvers."""


class ConvergenceError(RuntimeError):
    """Indicates that an implicit step did not converge."""


FixedStepperType = Callable[[NumericArray, float, int, Any], float]
AdaptiveStepperType = Callable[
    [NumericArray, float, float, float, OnlineStatistics | None, Any],
    tuple[float, float, int],
]


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
    _backend_name: BackendType | Literal["auto"]
    __backend_obj: BackendBase | None

    def __init__(
        self,
        pde: PDEBase,
        *,
        backend: BackendBase | BackendType | Literal["auto"] = "auto",
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            backend (str or :class:`~pde.backends.base.BackendBase`):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.
        """
        self.pde = pde
        self.info: dict[str, Any] = {"class": self.__class__.__name__}
        if self.pde:
            self.info["pde_class"] = self.pde.__class__.__name__
        if isinstance(backend, str):
            self._backend_name = backend
            self.__backend_obj = None
        else:
            # assume that `backend` is of type BackendBase
            self._backend_name = backend.name  # type: ignore
            self.__backend_obj = backend

    def __init_subclass__(cls, **kwargs):
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)

        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

        # register all subclasses to reconstruct them later
        if not isabstract(cls):
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}", stacklevel=2)
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
                for solver in sorted(cls._subclasses)
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
        return sorted(cls._subclasses)

    @property
    def backend(self) -> BackendType | Literal["auto"]:
        """str: The name of the backend used for this solver."""
        return self._backend_name

    @backend.setter
    def backend(self, value: BackendBase | BackendType) -> None:
        """Sets a new backend for the solver.

        This setter tries to ensure consistency and make sure that backends are not
        changed after the object has been loaded (i.e., after _backend_obj has been
        accessed). The method also raises a warning when the backend is changed (except
        if it was set to `auto` before). This allows solvers to react flexibly to
        changes in the backend, e.g., demanded by the PDE implementation.

        Args:
            value:
                The backend object or name
        """
        # determine the name of the new backend
        if isinstance(value, str):
            new_backend = value
        else:
            # assume value is of type BackendBase
            new_backend = value.name  # type: ignore

        # check whether the new name contradicts the old backend name
        if self._backend_name in {"auto", new_backend}:
            pass  # nothing to do
        else:
            self._logger.warning(
                "Changing the backend of the solver from `%s` to `%s`",
                self._backend_name,
                new_backend,
            )

        # check whether the new backend contradicts the old backend object
        if self.__backend_obj is not None and self.__backend_obj.name != new_backend:
            msg = "Tried changing the loaded backend of the solver from `%s` to `%s`"
            raise TypeError(
                msg,
                self.__backend_obj.name,
                new_backend,
            )

        # set the new backend
        self._backend_name = new_backend
        if not isinstance(value, str):
            self.__backend_obj = value

    @property
    def _backend_obj(self) -> BackendBase:
        """:class:`~pde.backends.base.BackendBase`: The backend used for this solver."""
        if self.__backend_obj is None:
            from ..backends import backends

            if self._backend_name == "auto":
                self._logger.warning(
                    "Automatic backend has not been selected yet. Choosing `numpy` "
                    "backend as a safe option."
                )
                self._backend_name = "numpy"  # conservative fall-back

            self.__backend_obj = backends[self._backend_name]

        return self.__backend_obj

    def _make_pde_rhs(
        self, state: TField
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Return a function for evaluating the right hand side of the PDE.

        This function also decides which backend will be used.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
            backend (str):
                Determines how the function is created. Accepted values are 'numpy' and
                'numba'. Alternatively, 'auto' lets the code pick the optimal backend.

        Returns:
            callable: Function determining the right hand side of the PDE
        """
        from ..backends import backends

        if self.backend == "auto":
            # try using the numba backend, if it implemented
            try:
                rhs = backends["numba"].make_pde_rhs(self.pde, state)
            except NotImplementedError:
                rhs = backends["numpy"].make_pde_rhs(self.pde, state)
                self.backend = "numpy"
            else:
                self.backend = "numba"
            return rhs

        # get a function evaluating the rhs of the PDE
        return self._backend_obj.make_pde_rhs(self.pde, state)

    def _make_error_synchronizer(
        self, backend: str = "numpy", *, operator: int | str = "MAX"
    ) -> Callable[[float], float]:
        """Return function that synchronizes errors between multiple processes.

        Args:
            backend (str):
                The backend to use for making the synchronizer
            operator (str or int):
                Flag determining how the value from multiple nodes is combined.
                Possible values include "MAX", "MIN", and "SUM".

        Returns:
            Function that can be used to synchronize errors across nodes
        """
        from ..backends import backends

        # deprecated on 2025-12-07
        warnings.warn(
            "`_make_error_synchronizer` is deprecated. Use `make_mpi_synchronizer` "
            "from an appropriate backend instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return backends[backend].make_mpi_synchronizer(operator=operator)

    def _make_post_step_hook(self, state: TField) -> StepperHook:
        """Create a function that calls the post-step hook of the PDE.

        A no-op function is returned if :attr:`SolverBase._use_post_step_hook` is
        `False` or the PDE does not provide :meth:`PDEBase.make_post_step_hook`.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.

        Returns:
            callable: The function that calls the post-step hook
        """
        post_step_hook: StepperHook | None = None

        if self._use_post_step_hook:
            if self._backend_name == "auto":
                msg = (
                    "Backend has not been chosen automatically, yet. Please specify a "
                    "backend explicitly."
                )

                raise ValueError(msg)
            try:
                # try to get hook function and initial data from PDE instance
                post_step_hook, self._post_step_data_init = (
                    self.pde.make_post_step_hook(state, backend=self._backend_name)
                )
                self._logger.info("Created post-step hook from PDE")

            except NotImplementedError:
                pass  # no hook function defined on the PDE

        if post_step_hook is None:
            # hook function is not necessary or was not supplied -> provide no-op

            def post_step_hook(
                state_data: NumericArray, t: float, post_step_data: NumericArray
            ):
                """Default hook function does nothing."""

            self._post_step_data_init = None
            self._logger.debug("No post-step hook defined")

        else:
            # ensure that the initial values is a mutable array
            self._post_step_data_init = np.array(self._post_step_data_init, copy=True)

        return post_step_hook  # type: ignore

    def _make_single_step_fixed_dt(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], None]:
        """Return a function doing a single step with a fixed time step.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.
        """
        msg = "Fixed stepper has not been defined"
        raise NotImplementedError(msg)

    def _make_fixed_stepper(self, state: TField, dt: float) -> FixedStepperType:
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

        def fixed_stepper(
            state_data: NumericArray, t_start: float, steps: int, post_step_data
        ) -> float:
            """Perform `steps` steps with fixed time steps."""
            for i in range(steps):
                # calculate the right hand side
                t = t_start + i * dt
                single_step(state_data, t)
                post_step_hook(state_data, t, post_step_data)

            return t + dt

        return fixed_stepper

    def make_stepper(
        self, state: TField, dt: float | None = None
    ) -> Callable[[TField, float, float], float]:
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
        fixed_stepper: FixedStepperType = self._backend_obj.make_inner_stepper(
            solver=self, stepper_style="fixed", state=state, dt=dt_float
        )

        self.info["dt"] = dt_float
        self.info["dt_adaptive"] = False
        self.info["steps"] = 0
        self.info["post_step_data"] = self._post_step_data_init
        self.info["stochastic"] = getattr(self.pde, "is_sde", False)
        # We don't access self.pde directly since we might want to reuse the solver
        # infrastructure for more general cases where a PDE is not defined.

        def wrapped_stepper(state: TField, t_start: float, t_end: float) -> float:
            """Advance `state` from `t_start` to `t_end` using fixed steps."""
            # retrieve last post_step_data and continue with this
            post_step_data = self.info["post_step_data"]
            # calculate number of steps that lead to an end time closest to t_end
            steps = max(1, round((t_end - t_start) / dt_float))
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
        backend: BackendType | Literal["auto"] = "auto",
        adaptive: bool = False,
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

    def _make_single_step_variable_dt(
        self, state: TField
    ) -> Callable[[NumericArray, float, float], NumericArray]:
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
        if self.pde.is_sde:
            msg = "Deterministic stepper does not support stochastic equations"
            raise RuntimeError(msg)

        rhs_pde = self._make_pde_rhs(state)

        def single_step(state_data: NumericArray, t: float, dt: float) -> NumericArray:
            """Basic implementation of Euler scheme."""
            return state_data + dt * rhs_pde(state_data, t)

        return self._backend_obj.compile_function(single_step)

    def _make_single_step_error_estimate(
        self, state: TField
    ) -> Callable[[NumericArray, float, float], tuple[NumericArray, float]]:
        """Make a stepper that also estimates the error.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
        """
        if getattr(self.pde, "is_sde", False):
            msg = "Cannot use adaptive stepper with stochastic equation"
            raise RuntimeError(msg)

        single_step = self._make_single_step_variable_dt(state)

        def single_step_error_estimate(
            state_data: NumericArray, t: float, dt: float
        ) -> tuple[NumericArray, float]:
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
        self, state: TField, *, adjust_dt: Callable[[float, float], float] | None = None
    ) -> AdaptiveStepperType:
        """Make an adaptive Euler stepper.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            adjust_dt (callable or None):
                A function that is used to adjust the time step. The function takes the
                current time step and a relative error and returns an adjusted time step

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
        if adjust_dt is None:
            adjust_dt = _make_dt_adjuster(self.dt_min, self.dt_max)
        tolerance = self.tolerance
        dt_min = self.dt_min

        def adaptive_stepper(
            state_data: NumericArray,
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

                # try two different step sizes to estimate errors
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

        self._logger.info("Initialized adaptive stepper")
        return adaptive_stepper

    def make_stepper(
        self, state: TField, dt: float | None = None
    ) -> Callable[[TField, float, float], float]:
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
            msg = "Cannot use adaptive stepper with stochastic equation"
            raise RuntimeError(msg)

        # Support `None` as a default value, so the controller can signal that
        # the solver should use a default time step.
        if dt is None:
            dt_float = self.dt_default
        else:
            dt_float = float(dt)  # explicit casting to help type checking

        # create stepper with fixed steps
        adaptive_stepper: AdaptiveStepperType = self._backend_obj.make_inner_stepper(
            solver=self, stepper_style="adaptive", state=state, dt=dt_float
        )

        self.info["dt"] = dt_float
        self.info["dt_adaptive"] = True
        self.info["steps"] = 0
        self.info["stochastic"] = getattr(self.pde, "is_sde", False)
        self.info["post_step_data"] = self._post_step_data_init

        def wrapped_stepper(state: TField, t_start: float, t_end: float) -> float:
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


def _make_dt_adjuster(dt_min: float, dt_max: float) -> Callable[[float, float], float]:
    """Return a function that can be used to adjust time steps.

    The returned function adjust_dt(dt, error_rel) adjusts the current time step
    `dt` based on the normalized error estimate `error_rel` with the goal of
    keeping `error_rel` close to 1.

    Behavior:
    - If the error is very small the time step is increased (up to a factor 4).
    - If the error is NaN the time step is reduced strongly.
    - Otherwise the time step is scaled according to error_rel**-0.2 with a
      conservative lower bound for the scaling factor.
    - The adjusted time step is clamped to the interval [dt_min, dt_max].
    - If the adjusted time step falls below dt_min a RuntimeError is raised.

    Args:
        dt_min (float): Minimal allowed time step.
        dt_max (float): Maximal allowed time step.

    Returns:
        Callable[[float, float], float]:
            Function that takes (dt, error_rel) and returns the adjusted dt.
    """
    dt_min_nan_err = f"Encountered NaN even though dt < {dt_min}"
    dt_min_err = f"Time step below {dt_min}"

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
            raise RuntimeError(dt_min_err)

        return dt

    return adjust_dt
