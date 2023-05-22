"""
Package that contains base classes for solvers

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import logging
import warnings
from abc import ABCMeta, abstractmethod
from inspect import isabstract
from typing import Any, Callable, Dict, List, Optional, Tuple, Type  # @UnusedImport

import numpy as np
from numba.extending import register_jitable

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from ..tools.math import OnlineStatistics
from ..tools.misc import classproperty
from ..tools.numba import is_jitted, jit


class SolverBase(metaclass=ABCMeta):
    """base class for solvers"""

    _subclasses: Dict[str, Type[SolverBase]] = {}  # all inheriting classes

    def __init__(self, pde: PDEBase):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
        """
        self.pde = pde
        self.info: Dict[str, Any] = {
            "class": self.__class__.__name__,
            "pde_class": self.pde.__class__.__name__,
        }
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
    def registered_solvers(cls) -> List[str]:  # @NoSelf
        """list of str: the names of the registered solvers"""
        return list(sorted(cls._subclasses.keys()))

    def _make_pde_rhs(
        self, state: FieldBase, backend: str = "auto"
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
        if self.pde.is_sde:
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
    ) -> Callable[[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
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

    @abstractmethod
    def make_stepper(
        self, state, dt: Optional[float] = None
    ) -> Callable[[FieldBase, float, float], float]:
        pass


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
        backend: str = "auto",
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
            tolerance (float):
                The error tolerance used in adaptive time stepping. This is used in
                adaptive time stepping to choose a time step which is small enough so
                the truncation error of a single step is below `tolerance`.
        """
        super().__init__(pde)
        self.backend = backend
        self.tolerance = tolerance

    def _make_error_synchronizer(self) -> Callable[[float], float]:
        """return helper function that synchronizes errors between multiple processes"""

        @register_jitable
        def synchronize_errors(error: float) -> float:
            return error

        return synchronize_errors  # type: ignore

    def _make_dt_adjuster(self) -> Callable[[float, float, float], float]:
        """return a function that can be used to adjust time steps"""
        dt_min = self.dt_min
        dt_min_err = f"Time step below {dt_min}"
        dt_max = self.dt_max

        def adjust_dt(dt: float, error_rel: float, t: float) -> float:
            """helper function that adjust the time step

            Args:
                dt (float): Current time step
                error_rel (float): Current (normalized) error estimate
                t (float): Current time point

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
                    raise RuntimeError("Encountered NaN during simulation")
                else:
                    raise RuntimeError(dt_min_err)

            return dt

        if self.backend == "numba":
            adjust_dt = jit(adjust_dt)

        return adjust_dt

    def _make_adaptive_stepper(
        self,
        state: FieldBase,
        paired_stepper: Callable[
            [np.ndarray, float, float], Tuple[np.ndarray, np.ndarray]
        ],
    ) -> Callable[
        [np.ndarray, float, float, float, Optional[OnlineStatistics]],
        Tuple[float, float, int, float],
    ]:
        """make an adaptive Euler stepper

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            paired_stepper (callable):
                (Compiled) function that advances the state in two different ways to be
                able to estimate the error.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        # obtain functions determining how the PDE is evolved
        modify_after_step = jit(self.pde.make_modify_after_step(state))

        # obtain auxiliary functions
        sync_errors = self._make_error_synchronizer()
        adjust_dt = self._make_dt_adjuster()
        tolerance = self.tolerance
        dt_min = self.dt_min
        compiled = self.backend == "numba"

        def stepper(
            state_data: np.ndarray,
            t_start: float,
            t_end: float,
            dt_init: float,
            dt_stats: Optional[OnlineStatistics] = None,
        ) -> Tuple[float, float, int, float]:
            """adaptive stepper that advances the state in time"""
            modifications = 0.0
            dt_opt = dt_init
            t = t_start
            steps = 0
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = max(min(dt_opt, t_end - t), dt_min)

                # two different steppings to estimate errors
                k1, k2 = paired_stepper(state_data, t, dt_step)

                # calculate maximal error
                if compiled:
                    error = 0.0
                    for i in range(state_data.size):
                        # max() has the weird behavior that `max(np.nan, 0)` is `np.nan`
                        # while `max(0, np.nan) == 0`. To propagate NaNs in the
                        # evaluation, we thus need to use the following order:
                        error = max(abs(k1.flat[i] - k2.flat[i]), error)
                else:
                    error = np.abs(k1 - k2).max()
                error_rel = error / tolerance  # normalize error to given tolerance

                # synchronize the error between all processes (if necessary)
                error_rel = sync_errors(error_rel)

                # do the step if the error is sufficiently small
                if error_rel <= 1:
                    steps += 1
                    t += dt_step
                    state_data[...] = k2
                    modifications += modify_after_step(state_data)
                    if dt_stats is not None:
                        dt_stats.add(dt_step)

                if t < t_end:
                    # adjust the time step and continue
                    dt_opt = adjust_dt(dt_step, error_rel, t)
                else:
                    break  # return to the controller

            return t, dt_opt, steps, modifications

        self._logger.info(f"Initialized adaptive stepper")
        return stepper
