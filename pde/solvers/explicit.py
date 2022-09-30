"""
Defines an explicit solver supporting various methods
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable, Optional, Tuple

import numba as nb
import numpy as np
from numba.extending import register_jitable

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from ..tools.math import OnlineStatistics
from ..tools.numba import jit
from .base import SolverBase


class ExplicitSolver(SolverBase):
    """class for solving partial differential equations explicitly"""

    name = "explicit"

    dt_min: float = 1e-10
    """float: minimal time step that the adaptive solver will use"""
    dt_max: float = 1e10
    """float: maximal time step that the adaptive solver will use"""

    def __init__(
        self,
        pde: PDEBase,
        scheme: str = "euler",
        *,
        backend: str = "auto",
        adaptive: bool = False,
        tolerance: float = 1e-4,
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The instance describing the pde that needs to be solved
            scheme (str):
                Defines the explicit scheme to use. Supported values are 'euler' and
                'runge-kutta' (or 'rk' for short).
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
        super().__init__(pde)
        self.scheme = scheme
        self.backend = backend
        self.adaptive = adaptive
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

    def _make_fixed_euler_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, int], Tuple[float, float]]:
        """make a simple Euler stepper with fixed time step

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, steps: int)`
        """
        # obtain post-step action function
        modify_after_step = jit(self.pde.make_modify_after_step(state))

        if self.pde.is_sde:
            # handle stochastic version of the pde
            rhs_sde = self._make_sde_rhs(state, backend=self.backend)

            def stepper(
                state_data: np.ndarray, t_start: float, steps: int
            ) -> Tuple[float, float]:
                """compiled inner loop for speed"""
                modifications = 0.0
                for i in range(steps):
                    # calculate the right hand side
                    t = t_start + i * dt
                    evolution_rate, noise_realization = rhs_sde(state_data, t)
                    state_data += dt * evolution_rate
                    if noise_realization is not None:
                        state_data += np.sqrt(dt) * noise_realization
                    modifications += modify_after_step(state_data)

                return t + dt, modifications

            self.info["stochastic"] = True
            self._logger.info(
                f"Initialized explicit Euler-Maruyama stepper with dt=%g", dt
            )

        else:
            # handle deterministic version of the pde
            rhs_pde = self._make_pde_rhs(state, backend=self.backend)

            def stepper(
                state_data: np.ndarray, t_start: float, steps: int
            ) -> Tuple[float, float]:
                """compiled inner loop for speed"""
                modifications = 0
                for i in range(steps):
                    # calculate the right hand side
                    t = t_start + i * dt
                    state_data += dt * rhs_pde(state_data, t)
                    modifications += modify_after_step(state_data)

                return t + dt, modifications

            self.info["stochastic"] = False
            self._logger.info(f"Initialized explicit Euler stepper with dt=%g", dt)

        return stepper

    def _make_adaptive_euler_stepper(
        self, state: FieldBase
    ) -> Callable[
        [np.ndarray, float, float, float, Optional[OnlineStatistics]],
        Tuple[float, float, int, float],
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
        if self.pde.is_sde:
            raise RuntimeError(
                "Cannot use adaptive Euler stepper with stochastic equation"
            )

        # obtain functions determining how the PDE is evolved
        rhs_pde = self._make_pde_rhs(state, backend=self.backend)
        modify_after_step = jit(self.pde.make_modify_after_step(state))

        # obtain auxiliary functions
        sync_errors = self._make_error_synchronizer()
        adjust_dt = self._make_dt_adjuster()
        tolerance = self.tolerance
        dt_min = self.dt_min

        def stepper(
            state_data: np.ndarray,
            t_start: float,
            t_end: float,
            dt_init: float,
            dt_stats: OnlineStatistics = None,
        ) -> Tuple[float, float, int, float]:
            """compiled inner loop for speed"""
            modifications = 0.0
            dt_opt = dt_init
            t = t_start
            calculate_rate = True  # flag stating whether to calculate rate for time t
            steps = 0
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = max(min(dt_opt, t_end - t), dt_min)

                if calculate_rate:
                    rate = rhs_pde(state_data, t)
                    calculate_rate = False
                # else: rate is reused from last (failed) iteration

                # single step with dt
                k1 = state_data + dt_step * rate

                # double step with half the time step
                k2 = state_data + 0.5 * dt_step * rate
                k2 += 0.5 * dt_step * rhs_pde(k2, t + 0.5 * dt_step)

                # calculate maximal error
                error = 0.0
                for i in range(state_data.size):
                    # max() has the weird behavior that `max(np.nan, 0)` is `np.nan`
                    # while `max(0, np.nan) == 0`. To propagate NaNs in the evaluation,
                    # we thus need to use the following order:
                    error = max(abs(k1.flat[i] - k2.flat[i]), error)
                error_rel = error / tolerance  # normalize error to given tolerance

                # synchronize the error between all processes (if necessary)
                error_rel = sync_errors(error_rel)

                # do the step if the error is sufficiently small
                if error_rel <= 1:
                    steps += 1
                    t += dt_step
                    state_data[...] = k2
                    modifications += modify_after_step(state_data)
                    calculate_rate = True
                    if dt_stats is not None:
                        dt_stats.add(dt_step)

                if t < t_end:
                    # adjust the time step and continue
                    dt_opt = adjust_dt(dt_step, error_rel, t)
                else:
                    break  # return to the controller

            return t, dt_opt, steps, modifications

        self._logger.info(f"Initialized adaptive Euler stepper")
        return stepper

    def _make_rk45_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, int], Tuple[float, float]]:
        """make a simple stepper for the explicit Runge-Kutta method of order 5(4)

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, steps: int)`
        """
        if self.pde.is_sde:
            raise RuntimeError(
                "Runge-Kutta stepper does not support stochastic equations"
            )
        self.info["stochastic"] = False

        # obtain functions determining how the PDE is evolved
        rhs = self._make_pde_rhs(state, backend=self.backend)
        modify_after_step = jit(self.pde.make_modify_after_step(state))

        def stepper(
            state_data: np.ndarray, t_start: float, steps: int
        ) -> Tuple[float, float]:
            """compiled inner loop for speed"""
            modifications = 0.0
            for i in range(steps):
                # calculate the right hand side
                t = t_start + i * dt

                # calculate the intermediate values in Runge-Kutta
                k1 = dt * rhs(state_data, t)
                k2 = dt * rhs(state_data + 0.5 * k1, t + 0.5 * dt)
                k3 = dt * rhs(state_data + 0.5 * k2, t + 0.5 * dt)
                k4 = dt * rhs(state_data + k3, t + dt)

                state_data += (k1 + 2 * k2 + 2 * k3 + k4) / 6
                modifications += modify_after_step(state_data)

            return t + dt, modifications

        self._logger.info(f"Initialized explicit Runge-Kutta-45 stepper with dt=%g", dt)
        return stepper

    def _make_rkf_stepper(
        self, state: FieldBase
    ) -> Callable[
        [np.ndarray, float, float, float, Optional[OnlineStatistics]],
        Tuple[float, float, int, float],
    ]:
        """make an adaptive stepper using the explicit Runge-Kutta-Fehlberg method

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        if self.pde.is_sde:
            raise RuntimeError(
                "Cannot use Runge-Kutta-Fehlberg stepper with stochastic equation"
            )

        # obtain functions determining how the PDE is evolved
        rhs = self._make_pde_rhs(state, backend=self.backend)
        modify_after_step = jit(self.pde.make_modify_after_step(state))
        self.info["stochastic"] = False

        # obtain auxiliary functions
        sync_errors = self._make_error_synchronizer()
        adjust_dt = self._make_dt_adjuster()
        tolerance = self.tolerance
        dt_min = self.dt_min

        # use Runge-Kutta-Fehlberg method
        # define coefficients for RK4(5), formula 2 Table III in Fehlberg
        a2 = 1 / 4
        a3 = 3 / 8
        a4 = 12 / 13
        a5 = 1.0
        a6 = 1 / 2

        b21 = 1 / 4
        b31 = 3 / 32
        b32 = 9 / 32
        b41 = 1932 / 2197
        b42 = -7200 / 2197
        b43 = 7296 / 2197
        b51 = 439 / 216
        b52 = -8.0
        b53 = 3680 / 513
        b54 = -845 / 4104
        b61 = -8 / 27
        b62 = 2.0
        b63 = -3544 / 2565
        b64 = 1859 / 4104
        b65 = -11 / 40

        r1 = 1 / 360
        # r2 = 0
        r3 = -128 / 4275
        r4 = -2197 / 75240
        r5 = 1 / 50
        r6 = 2 / 55

        c1 = 25 / 216
        # c2 = 0
        c3 = 1408 / 2565
        c4 = 2197 / 4104
        c5 = -1 / 5

        def stepper(
            state_data: np.ndarray,
            t_start: float,
            t_end: float,
            dt_init: float,
            dt_stats: OnlineStatistics = None,
        ) -> Tuple[float, float, int, float]:
            """compiled inner loop for speed"""
            modifications = 0.0
            dt_opt = dt_init
            t = t_start
            steps = 0
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = max(min(dt_opt, t_end - t), dt_min)

                # do the six intermediate steps
                k1 = dt_step * rhs(state_data, t)
                k2 = dt_step * rhs(state_data + b21 * k1, t + a2 * dt_step)
                k3 = dt_step * rhs(state_data + b31 * k1 + b32 * k2, t + a3 * dt_step)
                k4 = dt_step * rhs(
                    state_data + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * dt_step
                )
                k5 = dt_step * rhs(
                    state_data + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4,
                    t + a5 * dt_step,
                )
                k6 = dt_step * rhs(
                    state_data + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5,
                    t + a6 * dt_step,
                )

                # estimate the maximal error
                error = 0.0
                for i in range(state_data.size):
                    error_local = abs(
                        r1 * k1.flat[i]
                        + r3 * k3.flat[i]
                        + r4 * k4.flat[i]
                        + r5 * k5.flat[i]
                        + r6 * k6.flat[i]
                    )
                    # max() has the weird behavior that `max(np.nan, 0)` is `np.nan`
                    # while `max(0, np.nan) == 0`. To propagate NaNs in the evaluation,
                    # we thus need to use the following order:
                    error = max(error_local, error)  # type: ignore
                error_rel = error / tolerance  # normalize error to given tolerance

                # synchronize the error between all processes (if necessary)
                error_rel = sync_errors(error_rel)

                # do the step if the error is sufficiently small
                if error_rel <= 1:
                    steps += 1
                    t += dt_step
                    state_data += c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
                    modifications += modify_after_step(state_data)
                    if dt_stats is not None:
                        dt_stats.add(dt_step)

                if t < t_end:
                    # adjust the time step and continue
                    dt_opt = adjust_dt(dt_step, error_rel, t)
                else:
                    break  # return to the controller

            return t, dt_opt, steps, modifications

        self._logger.info(f"Initialized adaptive Runge-Kutta-Fehlberg stepper")
        return stepper

    def _make_fixed_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, int], Tuple[float, float]]:
        """return a stepper function using an explicit scheme with fixed time steps

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.
        """
        if self.scheme == "euler":
            fixed_stepper = self._make_fixed_euler_stepper(state, dt)
        elif self.scheme in {"runge-kutta", "rk", "rk45"}:
            fixed_stepper = self._make_rk45_stepper(state, dt)
        else:
            raise ValueError(f"Explicit scheme `{self.scheme}` is not supported")

        if self.backend == "numba":
            # compile inner stepper
            sig_fixed = (nb.typeof(state.data), nb.double, nb.int_)
            fixed_stepper = jit(sig_fixed)(fixed_stepper)

        self.info["dt_adaptive"] = False
        return fixed_stepper

    def _make_adaptive_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[
        [np.ndarray, float, float, float, OnlineStatistics],
        Tuple[float, float, int, float],
    ]:
        """return a stepper function using an explicit scheme with fixed time steps

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Initial time step of the adaptive explicit stepping
        """
        if self.pde.is_sde:
            raise NotImplementedError(
                "Adaptive stochastic stepping is not implemented. Use a fixed time "
                "step instead."
            )
        self.info["stochastic"] = False

        if self.scheme == "euler":
            adaptive_stepper = self._make_adaptive_euler_stepper(state)
        elif self.scheme in {"runge-kutta", "rk", "rk45"}:
            adaptive_stepper = self._make_rkf_stepper(state)
        else:
            raise ValueError(
                f"Explicit adaptive scheme `{self.scheme}` is not supported"
            )

        if self.backend == "numba":
            # compile inner stepper
            sig_adaptive = (
                nb.typeof(state.data),
                nb.double,
                nb.double,
                nb.double,
                nb.typeof(self.info["dt_statistics"]),
            )
            adaptive_stepper = jit(sig_adaptive)(adaptive_stepper)

        self.info["dt_adaptive"] = True
        return adaptive_stepper

    def make_stepper(
        self, state: FieldBase, dt: float = None
    ) -> Callable[[FieldBase, float, float], float]:
        """return a stepper function using an explicit scheme

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping. If `None`, this solver specifies
                1e-3 as a default value.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        # support `None` as a default value, so the controller can signal that
        # the solver should use a default time step
        if dt is None:
            dt = 1e-3
            if not self.adaptive:
                self._logger.warning(
                    "Explicit stepper with a fixed time step did not receive any "
                    f"initial value for `dt`. Using dt={dt}, but specifying a value or "
                    "enabling adaptive stepping is advisable."
                )
        dt_float = float(dt)  # explicit casting to help type checking

        self.info["dt"] = dt_float
        self.info["steps"] = 0
        self.info["scheme"] = self.scheme
        self.info["state_modifications"] = 0.0

        if self.adaptive:
            # create stepper with adaptive steps
            self.info["dt_statistics"] = OnlineStatistics()
            adaptive_stepper = self._make_adaptive_stepper(state, dt_float)

            def wrapped_stepper(
                state: FieldBase, t_start: float, t_end: float
            ) -> float:
                """advance `state` from `t_start` to `t_end` using adaptive steps"""
                nonlocal dt_float  # `dt_float` stores value for the next call

                t_last, dt_float, steps, modifications = adaptive_stepper(
                    state.data, t_start, t_end, dt_float, self.info["dt_statistics"]
                )
                self.info["steps"] += steps
                self.info["state_modifications"] += modifications
                return t_last

        else:
            # create stepper with fixed steps
            fixed_stepper = self._make_fixed_stepper(state, dt_float)

            def wrapped_stepper(
                state: FieldBase, t_start: float, t_end: float
            ) -> float:
                """advance `state` from `t_start` to `t_end` using fixed steps"""
                # calculate number of steps (which is at least 1)
                steps = max(1, int(np.ceil((t_end - t_start) / dt_float)))
                t_last, modifications = fixed_stepper(state.data, t_start, steps)
                self.info["steps"] += steps
                self.info["state_modifications"] += modifications
                return t_last

        return wrapped_stepper
