"""
Defines an explicit solver supporting various methods
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable, Tuple

import numpy as np

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from ..tools.numba import jit
from .base import SolverBase


class ExplicitSolver(SolverBase):
    """class for solving partial differential equations explicitly"""

    name = "explicit"

    dt_min = 1e-10
    dt_max = 1e10

    def __init__(
        self,
        pde: PDEBase,
        scheme: str = "euler",
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
            self.info["adaptive"] = False
            self._logger.info(f"Initialized explicit Euler stepper with dt=%g", dt)

        return stepper

    def _make_adaptive_euler_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, float, float], Tuple[float, float, int, float]]:
        """make an adaptive Euler stepper

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Initial time step of the explicit stepping.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        if self.pde.is_sde:
            raise RuntimeError(
                "Cannot use adaptive Euler stepper with stochastic equation"
            )

        # obtain post-step action function
        modify_after_step = jit(self.pde.make_modify_after_step(state))

        # handle deterministic version of the pde using an adaptive stepper
        rhs_pde = self._make_pde_rhs(state, backend=self.backend)
        tolerance = self.tolerance
        self.info["stochastic"] = False
        self.info["adaptive"] = True

        dt_min = self.dt_min
        dt_min_err = f"Time step below {dt_min}"
        dt_max = self.dt_max

        def stepper(
            state_data: np.ndarray, t_start: float, t_end: float, dt_init: float
        ) -> Tuple[float, float, int, float]:
            """compiled inner loop for speed"""
            modifications = 0.0
            dt = dt_init
            t = t_start
            steps = 0
            while t < t_end:
                # single step with dt
                rate = rhs_pde(state_data, t)
                k1 = state_data + dt * rate

                # double step with half the dt
                k2 = state_data + 0.5 * dt * rate
                k2 += 0.5 * dt * rhs_pde(k2, t + 0.5 * dt)

                # calculate maximal error
                error = 0.0
                for i in range(state_data.size):
                    error = max(error, abs(k1.flat[i] - k2.flat[i]))
                error *= dt  # error estimate should be independent of magnitude of dt

                # do the step if the error is sufficiently small
                if error <= tolerance:
                    t += dt
                    steps += 1
                    state_data[:] = k2
                    modifications += modify_after_step(state_data)

                # adjust the time step
                if error < 1e-8:
                    dt *= 4.0  # maximal increase in dt
                else:
                    dt *= min(max(0.9 * (tolerance / error) ** 0.2, 0.1), 4.0)
                if dt > dt_max:
                    dt = dt_max
                elif dt < dt_min:
                    raise RuntimeError(dt_min_err)

            return t, dt, steps, modifications

        self._logger.info(f"Initialized adaptive Euler stepper with dt_0={dt}")
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
                "Cannot use Runge-Kutta stepper with stochastic equation"
            )

        rhs = self._make_pde_rhs(state, backend=self.backend)
        self.info["stochastic"] = False
        self.info["adaptive"] = False

        # obtain post-step action function
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
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, float, float], Tuple[float, float, int, float]]:
        """make an adaptive stepper using the explicit Runge-Kutta-Fehlberg method

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Initial time step of the explicit stepping.

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        if self.pde.is_sde:
            raise RuntimeError(
                "Cannot use Runge-Kutta-Fehlberg stepper with stochastic equation"
            )

        rhs = self._make_pde_rhs(state, backend=self.backend)
        self.info["stochastic"] = False
        self.info["adaptive"] = True

        # obtain post-step action function
        modify_after_step = jit(self.pde.make_modify_after_step(state))

        # use Runge-Kutta-Fehlberg method
        tolerance = self.tolerance
        dt_min = self.dt_min
        dt_min_err = f"Time step below {dt_min}"
        dt_max = self.dt_max

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
            state_data: np.ndarray, t_start: float, t_end: float, dt_init: float
        ) -> Tuple[float, float, int, float]:
            """compiled inner loop for speed"""
            modifications = 0.0
            dt = dt_init
            t = t_start
            steps = 0
            while t < t_end:
                steps += 1

                # do the six intermediate steps
                k1 = dt * rhs(state_data, t)
                k2 = dt * rhs(state_data + b21 * k1, t + a2 * dt)
                k3 = dt * rhs(state_data + b31 * k1 + b32 * k2, t + a3 * dt)
                k4 = dt * rhs(state_data + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * dt)
                k5 = dt * rhs(
                    state_data + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, t + a5 * dt
                )
                k6 = dt * rhs(
                    state_data + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5,
                    t + a6 * dt,
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
                    error = max(error, error_local)  # type: ignore

                # do the step if the error is sufficiently small
                if error <= tolerance:
                    t += dt
                    state_data += c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
                    modifications += modify_after_step(state_data)

                # adjust the time step
                if error < 1e-8:
                    dt *= 4.0  # maximal increase in dt
                else:
                    dt *= min(max(0.9 * (tolerance / error) ** 0.2, 0.1), 4.0)
                if dt > dt_max:
                    dt = dt_max
                elif dt < dt_min:
                    raise RuntimeError(dt_min_err)

            return t, dt, steps, modifications

        self._logger.info(
            f"Initialized adaptive Runge-Kutta-Fehlberg stepper with initial dt={dt}"
        )
        return stepper

    def make_stepper(
        self, state: FieldBase, dt=None
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

        self.info["dt"] = dt
        self.info["steps"] = 0
        self.info["scheme"] = self.scheme
        self.info["state_modifications"] = 0.0

        if self.pde.is_sde and self.adaptive:
            self._logger.warning(
                "Adaptive stochastic stepping is not implemented. Using a fixed time "
                "step instead."
            )

        if self.adaptive and not self.pde.is_sde:
            # create stepper with adaptive steps
            self.info["dt_adaptive"] = True
            self.info["dt_last"] = dt  # store the time step between calls

            if self.scheme == "euler":
                adaptive_stepper = self._make_adaptive_euler_stepper(state, dt)
            elif self.scheme in {"runge-kutta", "rk", "rk45"}:
                adaptive_stepper = self._make_rkf_stepper(state, dt)
            else:
                raise ValueError(
                    f"Explicit adaptive scheme `{self.scheme}` is not supported"
                )

            if self.info["backend"] == "numba":
                adaptive_stepper = jit(adaptive_stepper)  # compile inner stepper

            def wrapped_stepper(
                state: FieldBase, t_start: float, t_end: float
            ) -> float:
                """advance `state` from `t_start` to `t_end` using adaptive steps"""
                t_last, dt, steps, modifications = adaptive_stepper(
                    state.data, t_start, t_end, self.info["dt_last"]
                )
                self.info["dt_last"] = dt
                self.info["steps"] += steps
                self.info["state_modifications"] += modifications
                return t_last

        else:
            # create stepper with fixed steps
            self.info["dt_adaptive"] = False

            if self.scheme == "euler":
                fixed_stepper = self._make_fixed_euler_stepper(state, dt)
            elif self.scheme in {"runge-kutta", "rk", "rk45"}:
                fixed_stepper = self._make_rk45_stepper(state, dt)
            else:
                raise ValueError(f"Explicit scheme `{self.scheme}` is not supported")

            if self.info["backend"] == "numba":
                fixed_stepper = jit(fixed_stepper)  # compile inner stepper

            def wrapped_stepper(
                state: FieldBase, t_start: float, t_end: float
            ) -> float:
                """advance `state` from `t_start` to `t_end` using fixed steps"""
                # calculate number of steps (which is at least 1)
                steps = max(1, int(np.ceil((t_end - t_start) / dt)))
                t_last, modifications = fixed_stepper(state.data, t_start, steps)
                self.info["steps"] += steps
                self.info["state_modifications"] += modifications
                return t_last

        return wrapped_stepper
