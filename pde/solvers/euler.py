"""Defines an explicit solver for the Euler scheme.

.. autosummary::
   :nosignatures:

   EulerSolver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..tools.math import OnlineStatistics
from ..tools.misc import get_array_namespace
from .base import AdaptiveSolverBase, _make_dt_adjuster

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..backends.base import BackendBase
    from ..pdes.base import PDEBase
    from ..tools.typing import InnerStepperType, NumericArray, TField


def _DUMMY_FUNCTION():
    """Dummy function that we need to define so numba can determine types correctly."""
    return 0


def _DUMMY_FUNCTION_2ARGS(state_data, t):
    """Dummy function that we need to define so numba can determine types correctly."""
    return state_data


class EulerSolver(AdaptiveSolverBase):
    """Explicit Euler solver."""

    name = "euler"

    def __init__(
        self,
        pde: PDEBase,
        *,
        backend: str | BackendBase = "auto",
        adaptive: bool = False,
        tolerance: float = 1e-4,
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            backend (str):
                The backend used for numerical operations
            adaptive (bool):
                Whether to use adaptive time stepping
            tolerance (float):
                Error tolerance for adaptive time stepping
        """
        super().__init__(pde, backend=backend, adaptive=adaptive, tolerance=tolerance)

    def _make_single_step_fixed_dt_stochastic(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Make a Euler-Maruyama single-step update with fixed time step.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.

        Returns:
            Function that updates the state by one time step. The function call
            signature is `(state_data: numpy.ndarray, t: float)`.
        """
        # create deterministic part
        rhs_pde = self.backend.make_pde_rhs(self.pde, state)

        # handle with first noise interface based on supplying the noise variance
        noise_drift_factor = self.pde._noise_drift_factor
        has_noise_drift_term = noise_drift_factor != 0
        if use_noise_variance := self.pde.use_noise_variance:
            noise_var = self.pde.make_noise_variance(  # type: ignore
                state, backend=self.backend, ret_diff=has_noise_drift_term
            )
            gaussian_noise = self.backend.make_gaussian_noise(state, rng=self.pde.rng)
        else:
            noise_var = _DUMMY_FUNCTION_2ARGS
            gaussian_noise = self.backend.compile_function(_DUMMY_FUNCTION)
        noise_var = self.backend.compile_function(noise_var)

        # handle with second noise interface based on supplying a realization
        if use_noise_realization := self.pde.use_noise_realization:
            rhs_noise = self.pde.make_noise_realization(state, backend=self.backend)  # type: ignore
        else:
            rhs_noise = _DUMMY_FUNCTION_2ARGS
        rhs_noise = self.backend.compile_function(rhs_noise)

        # noise increment scales with square root of time step
        dt_sqrt = np.sqrt(dt)

        def single_step(state_data: NumericArray, t: float) -> NumericArray:
            """Perform a single Euler-Maruyama step."""
            # support any backend following Python Array API
            nx = get_array_namespace(state_data)

            # evaluate field-dependent terms first without modifying fields
            evolution_rate = rhs_pde(state_data, t)
            if use_noise_variance:
                if has_noise_drift_term:
                    noise_var_field, noise_var_diff_field = noise_var(state_data, t)
                else:
                    noise_var_field = noise_var(state_data, t)

            # handle second noise interface first so it uses the unchanged field
            if use_noise_realization:
                noise_realization = rhs_noise(state_data, t)
                if noise_realization is not None:
                    state_data += dt_sqrt * noise_realization

            # apply the deterministic part and the additive noise
            state_data += dt * evolution_rate

            if use_noise_variance:
                dW = gaussian_noise()
                state_data += dt_sqrt * nx.sqrt(noise_var_field) * dW

                # add a drift term if the interpretation is not Itô
                if has_noise_drift_term:
                    state_data += 0.5 * dt * noise_drift_factor * noise_var_diff_field

            return state_data

        self._logger.info(
            "Initialize explicit Euler-Maruyama single-step update with dt=%g", dt
        )
        return single_step

    def _make_single_step_fixed_dt(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Make a simple Euler single-step update with fixed time step.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping.

        Returns:
            Function that updates the state by one time step. The function call
            signature is `(state_data: numpy.ndarray, t: float)`.
        """
        if self.pde.is_sde:
            # handle stochastic version of the pde
            return self._make_single_step_fixed_dt_stochastic(state, dt)

        # handle deterministic version of the pde
        rhs_pde = self.backend.make_pde_rhs(self.pde, state)

        def single_step(state_data: NumericArray, t: float) -> NumericArray:
            """Perform a single Euler step."""
            state_data += dt * rhs_pde(state_data, t)
            return state_data

        self._logger.info("Initialize explicit Euler single-step update with dt=%g", dt)

        return single_step

    def _make_inner_stepper(self, state: TField) -> InnerStepperType:
        """Create the executable Euler stepping function for this solver.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        if not self.adaptive:
            # create a stepping function with fixed steps
            return super()._make_inner_stepper(state)

        # General comment: We implement the full adaptive scheme here instead of just
        # defining `_make_single_step_error_estimate` to do some optimizations. In
        # particular, we reuse the calculated right hand side in cases where the step
        # was not successful.
        if getattr(self.pde, "is_sde", False):
            msg = "Cannot use adaptive stepping with stochastic equation"
            raise RuntimeError(msg)

        # obtain functions determining how the PDE is evolved
        rhs_pde = self.backend.make_pde_rhs(self.pde, state)
        # if post_step_hook is None:
        post_step_hook = self._make_post_step_hook(state)

        # obtain auxiliary functions
        sync_errors = self.backend.make_mpi_synchronizer(operator="MAX")
        # if adjust_dt is None:
        adjust_dt = _make_dt_adjuster(self.dt_min, self.dt_max)
        tolerance = self.tolerance
        dt_min = self.dt_min

        # add extra information
        self.info["dt_adaptive"] = self.adaptive
        self.info["dt_statistics"] = OnlineStatistics()

        def adaptive_stepper(
            state_data: NumericArray, t_start: float, t_end: float
        ) -> float:
            """Adaptive stepping function that advances the state in time."""
            state_cur = state_data
            dt_opt = self.info["dt"]  # time step from last step
            rate = rhs_pde(state_cur, t_start)  # calculate initial rate

            steps = 0
            t = t_start
            while True:
                # use a smaller (but not too small) time step if close to t_end
                dt_step = max(min(dt_opt, t_end - t), dt_min)

                # do single step with dt
                step_large = state_cur + dt_step * rate
                # do double step with half the time step
                step_small = state_cur + 0.5 * dt_step * rate

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
                        state_cur, self.info["post_step_data"] = post_step_hook(
                            step_small, t, self.info["post_step_data"]
                        )
                        if self.info.get("dt_statistics"):
                            self.info["dt_statistics"].add(dt_step)

                if t < t_end:
                    # adjust the time step and continue
                    dt_opt = adjust_dt(dt_step, error_rel)
                else:
                    break  # return to the controller

            self.info["dt"] = dt_opt  # save last optimal time step
            self.info["steps"] += steps
            state_data[:] = state_cur
            return t

        self._logger.info("Initialize adaptive Euler stepping function")
        return adaptive_stepper


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
            **kwargs:
                Additional arguments such as `backend`, `adaptive`, and `tolerance` that
                are forwarded to the chosen solver class.
        """
        # deprecated since 2025-11-01
        warnings.warn(
            "`ExplicitSolver` is deprecated. Use `EulerSolver` or `RungeKuttaSolver`.",
            stacklevel=2,
        )
        if scheme == "euler":
            return EulerSolver(pde=pde, **kwargs)
        if scheme in {"rk", "rk45", "runge-kutta"}:
            from .runge_kutta import RungeKuttaSolver

            return RungeKuttaSolver(pde=pde, **kwargs)
        msg = f"Scheme `{scheme}` is not supported"
        raise ValueError(msg)
