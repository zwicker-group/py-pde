"""Defines an explicit solver supporting various methods.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from .base import AdaptiveSolverBase, AdaptiveStepperType, _make_dt_adjuster

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..pdes.base import PDEBase
    from ..tools.math import OnlineStatistics
    from ..tools.typing import NumericArray, StepperHook, TField


class EulerSolver(AdaptiveSolverBase):
    """Explicit Euler solver."""

    name = "euler"

    def _make_single_step_fixed_dt(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], None]:
        """Make a simple Euler stepper with fixed time step.

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
            # handle stochastic version of the pde
            self.info["scheme"] = "euler-maruyama"
            rhs_pde = self._make_pde_rhs(state)
            rhs_noise = self.pde.make_noise_realization(state, backend=self.backend)  # type: ignore

            def stepper(state_data: NumericArray, t: float) -> None:
                """Perform a single Euler-Maruyama step."""
                evolution_rate = rhs_pde(state_data, t)
                noise_realization = rhs_noise(state_data, t)
                state_data += dt * evolution_rate
                if noise_realization is not None:
                    state_data += np.sqrt(dt) * noise_realization

            self._logger.info("Init explicit Euler-Maruyama stepper with dt=%g", dt)

        else:
            # handle deterministic version of the pde
            if self.pde.is_sde:
                msg = "Deterministic Euler stepper doesn't support stochastic equations"
                raise RuntimeError(msg)

            self.info["scheme"] = "euler"
            rhs_pde = self._make_pde_rhs(state)

            def stepper(state_data: NumericArray, t: float) -> None:
                """Perform a single Euler step."""
                state_data += dt * rhs_pde(state_data, t)

            self._logger.info("Init explicit Euler stepper with dt=%g", dt)

        return stepper

    def _make_adaptive_stepper(
        self,
        state: TField,
        *,
        post_step_hook: StepperHook | None = None,
        adjust_dt: Callable[[float, float], float] | None = None,
    ) -> AdaptiveStepperType:
        """Make an adaptive Euler stepper.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            post_step_hook (callable or None):
                A function that runs the post_step_hook
            adjust_dt (callable or None):
                A function that is used to adjust the time step. The function takes the
                current time step and a relative error and returns an adjusted time step

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        # General comment: We implement the full adaptive scheme here instead of just
        # defining `_make_single_step_error_estimate` to do some optimizations. In
        # particular, we reuse the calculated right hand side in cases where the step
        # was not successful.
        if self.pde.is_sde:
            msg = "Deterministic adaptive stepper does not support stochastic equations"
            raise RuntimeError(msg)

        # obtain functions determining how the PDE is evolved
        rhs_pde = self._make_pde_rhs(state)
        if post_step_hook is None:
            post_step_hook = self._make_post_step_hook(state)

        # obtain auxiliary functions
        sync_errors = self._backend_obj.make_mpi_synchronizer(operator="MAX")
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
            rate = rhs_pde(state_data, t_start)  # calculate initial rate

            steps = 0
            t = t_start
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = max(min(dt_opt, t_end - t), dt_min)

                # do single step with dt
                step_large = state_data + dt_step * rate

                # do double step with half the time step
                step_small = state_data + 0.5 * dt_step * rate

                try:
                    # calculate rate at the midpoint of the double step
                    rate_midpoint = rhs_pde(step_small, t + 0.5 * dt_step)
                except Exception:
                    # an exception likely signals that rate could not be calculated
                    error_rel = np.nan
                else:
                    # advance to end of double step
                    step_small += 0.5 * dt_step * rate_midpoint

                    # calculate maximal error
                    error = np.abs(step_large - step_small).max()
                    error_rel = error / tolerance  # normalize error to given tolerance

                # synchronize the error between all processes (necessary for MPI)
                error_rel = sync_errors(error_rel)

                if error_rel <= 1:  # error is sufficiently small
                    try:
                        # calculating the rate at putative new step
                        rate = rhs_pde(step_small, t)
                    except Exception:
                        # calculating the rate failed => retry with smaller dt
                        error_rel = np.nan
                    else:
                        # everything worked => do the step
                        steps += 1
                        t += dt_step
                        state_data[...] = step_small
                        post_step_hook(state_data, t, post_step_data)
                        if dt_stats is not None:
                            dt_stats.add(dt_step)

                if t < t_end:
                    # adjust the time step and continue
                    dt_opt = adjust_dt(dt_step, error_rel)
                else:
                    break  # return to the controller

            return t, dt_opt, steps

        self._logger.info("Init adaptive Euler stepper")
        return adaptive_stepper


class RungeKuttaSolver(AdaptiveSolverBase):
    """Explicit Runge-Kutta PDE solver of order 5(4)."""

    name = "runge-kutta"

    def _make_single_step_fixed_dt(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], None]:
        """Make function doing a single explicit Runge-Kutta step of order 5(4)

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
            msg = "Deterministic Runge-Kutta does not support stochastic equations"
            raise RuntimeError(msg)

        # obtain functions determining how the PDE is evolved
        rhs = self._make_pde_rhs(state)

        def stepper(state_data: NumericArray, t: float) -> None:
            """Compiled inner loop for speed."""
            # calculate the intermediate values in Runge-Kutta
            k1 = dt * rhs(state_data, t)
            k2 = dt * rhs(state_data + 0.5 * k1, t + 0.5 * dt)
            k3 = dt * rhs(state_data + 0.5 * k2, t + 0.5 * dt)
            k4 = dt * rhs(state_data + k3, t + dt)

            state_data += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        self._logger.info("Init explicit Runge-Kutta-45 stepper with dt=%g", dt)
        return stepper

    def _make_single_step_error_estimate(
        self, state: TField
    ) -> Callable[[NumericArray, float, float], tuple[NumericArray, float]]:
        """Make an adaptive stepper using the explicit Runge-Kutta-Fehlberg method.

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
            msg = "Deterministic Runge-Kutta does not support stochastic equations"
            raise RuntimeError(msg)

        # obtain functions determining how the PDE is evolved
        rhs = self._make_pde_rhs(state)

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
            state_data: NumericArray, t: float, dt: float
        ) -> tuple[NumericArray, float]:
            """Basic stepper to estimate error."""
            # do the six intermediate steps
            k1 = dt * rhs(state_data, t)
            k2 = dt * rhs(state_data + b21 * k1, t + a2 * dt)
            k3 = dt * rhs(state_data + b31 * k1 + b32 * k2, t + a3 * dt)
            k4 = dt * rhs(state_data + b41 * k1 + b42 * k2 + b43 * k3, t + a4 * dt)
            k5 = dt * rhs(
                state_data + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4,
                t + a5 * dt,
            )
            k6 = dt * rhs(
                state_data + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5,
                t + a6 * dt,
            )

            # estimate the maximal error
            error_local = r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6
            error = np.abs(error_local).max()

            state_new = state_data + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
            return state_new, error

        self._logger.info("Init adaptive Runge-Kutta-Fehlberg stepper")
        return stepper


class ExplicitSolver(AdaptiveSolverBase):
    """Various explicit PDE solvers."""

    name = "explicit"

    def __new__(
        cls,
        pde: PDEBase,
        scheme: Literal["euler", "runge-kutta", "rk", "rk45"] = "euler",
        **kwargs,
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
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
        # deprecated since 2025-11-01
        warnings.warn(
            "`ExplicitSolver` is deprecated. Use `EulerSolver` or `RungeKuttaSolver`.",
            stacklevel=2,
        )
        if scheme == "euler":
            return EulerSolver(pde=pde, **kwargs)
        if scheme in {"rk", "rk45", "runge-kutta"}:
            return RungeKuttaSolver(pde=pde, **kwargs)
        msg = f"Scheme `{scheme}` is not supported"
        raise ValueError(msg)
