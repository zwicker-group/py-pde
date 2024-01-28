"""
Package that contains base classes for solvers.

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
from ..tools.typing import BackendType


class ConvergenceError(RuntimeError):
    """indicates that an implicit step did not converge"""


class SolverBase(metaclass=ABCMeta):
    """base class for PDE solvers"""

    dt_default: float = 1e-3
    """float: default time step used if no time step was specified"""

    _modify_state_after_step: bool = True
    """bool: flag choosing whether the `modify_after_step` hook of the PDE is called"""

    _subclasses: dict[str, type[SolverBase]] = {}
    """dict: dictionary of all inheriting classes"""

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
        self._logger = logging.getLogger(self.__class__.__name__)

    def __init_subclass__(cls, **kwargs):  # @NoSelf
        """register all subclassess to reconstruct them later"""
        super().__init_subclass__(**kwargs)
        if not isabstract(cls):
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}")
            cls._subclasses[cls.__name__] = cls
        if hasattr(cls, "name") and cls.name:
            if cls.name in cls._subclasses:
                logging.warning(f"Solver with name {cls.name} is already registered")
            cls._subclasses[cls.name] = cls

    @classmethod
    def from_name(cls, name: str, pde: PDEBase, **kwargs) -> SolverBase:
        r"""create solver class based on its name

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
            )

        return solver_class(pde, **kwargs)

    @classproperty
    def registered_solvers(cls) -> list[str]:  # @NoSelf
        """list of str: the names of the registered solvers"""
        return list(sorted(cls._subclasses.keys()))

    @property
    def _compiled(self) -> bool:
        """bool: indicates whether functions need to be compiled"""
        return (
            self.backend == "numba" and not nb.config.DISABLE_JIT
        )  # @UndefinedVariable

    def _make_modify_after_step(
        self, state: FieldBase
    ) -> Callable[[np.ndarray], float]:
        """create a function that modifies a state after each step

        A noop function will be returned if `_modify_state_after_step` is `False`,

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted.
        """
        if self._modify_state_after_step:
            modify_after_step = jit(self.pde.make_modify_after_step(state))

        else:

            def modify_after_step(state_data: np.ndarray) -> float:
                return 0

        if self._compiled:
            sig_modify = (nb.typeof(state.data),)
            modify_after_step = jit(sig_modify)(modify_after_step)

        return modify_after_step  # type: ignore

    def _make_pde_rhs(
        self, state: FieldBase, backend: BackendType = "auto"
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """obtain a function for evaluating the right hand side

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
        if getattr(self.pde, "is_sde"):
            raise RuntimeError(
                f"Cannot create a deterministic stepper for a stochastic equation"
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
        """obtain a function for evaluating the right hand side

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
        """return a function doing a single step with a fixed time step

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
    ) -> Callable[[np.ndarray, float, int], tuple[float, float]]:
        """return a stepper function using an explicit scheme with fixed time steps

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.
        """
        single_step = self._make_single_step_fixed_dt(state, dt)
        modify_state_after_step = self._modify_state_after_step
        modify_after_step = self._make_modify_after_step(state)

        if self._compiled:
            sig_single_step = (nb.typeof(state.data), nb.double)
            single_step = jit(sig_single_step)(single_step)

        def fixed_stepper(
            state_data: np.ndarray, t_start: float, steps: int
        ) -> tuple[float, float]:
            """perform `steps` steps with fixed time steps"""
            modifications = 0.0
            for i in range(steps):
                # calculate the right hand side
                t = t_start + i * dt
                single_step(state_data, t)
                if modify_state_after_step:
                    modifications += modify_after_step(state_data)

            return t + dt, modifications

        if self._compiled:
            sig_fixed = (nb.typeof(state.data), nb.double, nb.int_)
            fixed_stepper = jit(sig_fixed)(fixed_stepper)

        return fixed_stepper

    def make_stepper(
        self, state: FieldBase, dt: float | None = None
    ) -> Callable[[FieldBase, float, float], float]:
        """return a stepper function using an explicit scheme

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
                f"initial value for `dt`. Using dt={dt}, but specifying a value or "
                "enabling adaptive stepping is advisable."
            )
        dt_float = float(dt)  # explicit casting to help type checking

        self.info["dt"] = dt_float
        self.info["steps"] = 0
        self.info["state_modifications"] = 0.0
        self.info["stochastic"] = getattr(self.pde, "is_sde", False)
        # we don't access self.pde directly since we might want to reuse the solver
        # infrastructure for more general cases where a PDE is not defined

        # create stepper with fixed steps
        fixed_stepper = self._make_fixed_stepper(state, dt_float)

        def wrapped_stepper(state: FieldBase, t_start: float, t_end: float) -> float:
            """advance `state` from `t_start` to `t_end` using fixed steps"""
            # calculate number of steps (which is at least 1)
            steps = max(1, int(np.ceil((t_end - t_start) / dt_float)))
            t_last, modifications = fixed_stepper(state.data, t_start, steps)
            self.info["steps"] += steps
            self.info["state_modifications"] += modifications
            return t_last

        return wrapped_stepper


class AdaptiveSolverBase(SolverBase):
    """base class for adaptive time steppers"""

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
                The instance describing the pde that needs to be solved
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

    def _make_error_synchronizer(self) -> Callable[[float], float]:
        """return helper function that synchronizes errors between multiple processes"""

        @register_jitable
        def synchronize_errors(error: float) -> float:
            return error

        return synchronize_errors  # type: ignore

    def _make_dt_adjuster(self) -> Callable[[float, float], float]:
        """return a function that can be used to adjust time steps"""
        dt_min = self.dt_min
        dt_min_nan_err = f"Encountered NaN even though dt < {dt_min}"
        dt_min_err = f"Time step below {dt_min}"
        dt_max = self.dt_max

        def adjust_dt(dt: float, error_rel: float) -> float:
            """helper function that adjust the time step

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
        """return a function doing a single step with a variable time step

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
            """basic implementation of Euler scheme"""
            return state_data + dt * rhs_pde(state_data, t)  # type: ignore

        return single_step

    def _make_single_step_error_estimate(
        self, state: FieldBase
    ) -> Callable[[np.ndarray, float, float], tuple[np.ndarray, float]]:
        """make a stepper that also estimates the error

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
        """
        if getattr(self.pde, "is_sde"):
            raise RuntimeError("Cannot use adaptive stepper with stochastic equation")

        single_step = self._make_single_step_variable_dt(state)
        if self._compiled:
            sig_single_step = (nb.typeof(state.data), nb.double, nb.double)
            single_step = jit(sig_single_step)(single_step)

        def single_step_error_estimate(
            state_data: np.ndarray, t: float, dt: float
        ) -> tuple[np.ndarray, float]:
            """basic stepper to estimate error"""
            # single step with dt
            k1 = single_step(state_data, t, dt)

            # double step with half the time step
            k2a = single_step(state_data, t, 0.5 * dt)
            k2 = single_step(k2a, t + 0.5 * dt, 0.5 * dt)

            # calculate maximal error
            error = np.abs(k1 - k2).max()

            return k2, error

        return single_step_error_estimate

    def _make_adaptive_stepper(self, state: FieldBase) -> Callable[
        [np.ndarray, float, float, float, OnlineStatistics | None],
        tuple[float, float, int, float],
    ]:
        """make an adaptive Euler stepper

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
        modify_after_step = self._make_modify_after_step(state)
        modify_state_after_step = self._modify_state_after_step
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
        ) -> tuple[float, float, int, float]:
            """adaptive stepper that advances the state in time"""
            modifications = 0.0
            dt_opt = dt_init
            t = t_start
            steps = 0
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = max(min(dt_opt, t_end - t), dt_min)

                # two different steppings to estimate errors
                new_state, error = single_step_error(state_data, t, dt_step)

                error_rel = error / tolerance  # normalize error to given tolerance
                # synchronize the error between all processes (if necessary)
                error_rel = sync_errors(error_rel)

                # do the step if the error is sufficiently small
                if error_rel <= 1:
                    steps += 1
                    t += dt_step
                    state_data[...] = new_state
                    if modify_state_after_step:
                        modifications += modify_after_step(state_data)
                    if dt_stats is not None:
                        dt_stats.add(dt_step)

                if t < t_end:
                    # adjust the time step and continue
                    dt_opt = adjust_dt(dt_step, error_rel)
                else:
                    break  # return to the controller

            return t, dt_opt, steps, modifications

        if self._compiled:
            # compile inner stepper
            sig_adaptive = (
                nb.typeof(state.data),
                nb.double,
                nb.double,
                nb.double,
                nb.typeof(self.info["dt_statistics"]),
            )
            adaptive_stepper = jit(sig_adaptive)(adaptive_stepper)

        self._logger.info(f"Initialized adaptive stepper")
        return adaptive_stepper

    def make_stepper(
        self, state: FieldBase, dt: float | None = None
    ) -> Callable[[FieldBase, float, float], float]:
        """return a stepper function using an explicit scheme

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

        # support `None` as a default value, so the controller can signal that
        # the solver should use a default time step
        if dt is None:
            dt_float = self.dt_default
        else:
            dt_float = float(dt)  # explicit casting to help type checking

        self.info["dt"] = dt_float
        self.info["dt_adaptive"] = True
        self.info["steps"] = 0
        self.info["stochastic"] = getattr(self.pde, "is_sde", False)
        self.info["state_modifications"] = 0.0

        # create stepper with adaptive steps
        self.info["dt_statistics"] = OnlineStatistics()
        adaptive_stepper = self._make_adaptive_stepper(state)

        def wrapped_stepper(state: FieldBase, t_start: float, t_end: float) -> float:
            """advance `state` from `t_start` to `t_end` using adaptive steps"""
            nonlocal dt_float  # `dt_float` stores value for the next call

            t_last, dt_float, steps, modifications = adaptive_stepper(
                state.data, t_start, t_end, dt_float, self.info["dt_statistics"]
            )
            self.info["steps"] += steps
            self.info["state_modifications"] += modifications
            return t_last

        return wrapped_stepper
