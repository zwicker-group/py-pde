"""
Defines an implicit Euler solver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

from typing import Callable

import numba as nb
import numpy as np

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from ..tools.typing import BackendType
from .base import ConvergenceError, SolverBase


class ImplicitSolver(SolverBase):
    """implicit (backward) Euler PDE solver"""

    name = "implicit"

    def __init__(
        self,
        pde: PDEBase,
        maxiter: int = 100,
        maxerror: float = 1e-4,
        backend: BackendType = "auto",
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The instance describing the pde that needs to be solved
            maxiter (int):
                The maximal number of iterations per step
            maxerror (float):
                The maximal error that is permitted in each step
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.
        """
        super().__init__(pde, backend=backend)
        self.maxiter = maxiter
        self.maxerror = maxerror

    def _make_single_step_fixed_dt_deterministic(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float], None]:
        """return a function doing a deterministic step with an implicit Euler scheme

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the implicit step
        """
        if self.pde.is_sde:
            raise RuntimeError("Cannot use implicit stepper with stochastic equation")

        self.info["function_evaluations"] = 0
        self.info["scheme"] = "implicit-euler"
        self.info["stochastic"] = False
        self.info["dt_adaptive"] = False

        rhs = self._make_pde_rhs(state, backend=self.backend)
        maxiter = int(self.maxiter)
        maxerror2 = self.maxerror**2

        # handle deterministic version of the pde
        def implicit_step(state_data: np.ndarray, t: float) -> None:
            """compiled inner loop for speed"""
            nfev = 0  # count function evaluations

            # save state at current time point t for stepping
            state_t = state_data.copy()

            # estimate state at next time point
            state_data[:] = state_t + dt * rhs(state_data, t)
            state_prev = np.empty_like(state_data)

            # fixed point iteration for improving state after dt
            for n in range(maxiter):
                state_prev[:] = state_data  # keep previous state to judge convergence
                # another interation to improve estimate
                state_data[:] = state_t + dt * rhs(state_data, t + dt)

                # calculate mean squared error to judge convergence
                err = 0.0
                for j in range(state_data.size):
                    diff = state_data.flat[j] - state_prev.flat[j]
                    err += (diff.conjugate() * diff).real
                err /= state_data.size

                if err < maxerror2:
                    # fix point iteration converged
                    break
            else:
                with nb.objmode:
                    self._logger.warning(
                        "Implicit Euler step did not converge after %d iterations "
                        "at t=%g (error=%g)",
                        maxiter,
                        t,
                        err,
                    )
                raise ConvergenceError("Implicit Euler step did not converge.")
            nfev += n + 1

        self._logger.info("Init implicit Euler stepper with dt=%g", dt)
        return implicit_step

    def _make_single_step_fixed_dt_stochastic(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float], None]:
        """return a function doing a step for a SDE with an implicit Euler scheme

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the implicit step
        """
        self.info["function_evaluations"] = 0
        self.info["scheme"] = "implicit-euler-maruyama"
        self.info["stochastic"] = True
        self.info["dt_adaptive"] = False

        rhs = self.pde.make_pde_rhs(state, backend=self.backend)  # type: ignore
        rhs_sde = self._make_sde_rhs(state, backend=self.backend)
        maxiter = int(self.maxiter)
        maxerror2 = self.maxerror**2

        # handle deterministic version of the pde
        def implicit_step(state_data: np.ndarray, t: float) -> None:
            """compiled inner loop for speed"""
            nfev = 0  # count function evaluations

            # save state at current time point t for stepping
            state_t = state_data.copy()

            # estimate state at next time point
            evolution_rate, noise_realization = rhs_sde(state_data, t)
            state_data[:] = state_t + dt * evolution_rate  # estimated state

            if noise_realization is not None:
                # add the noise to the reference state at the current time point and
                # adept the state at the next time point iteratively below
                state_t += np.sqrt(dt) * noise_realization

            state_prev = np.empty_like(state_data)

            # fixed point iteration for improving state after dt
            for n in range(maxiter):
                state_prev[:] = state_data  # keep previous state to judge convergence
                # another interation to improve estimate
                state_data[:] = state_t + dt * rhs(state_data, t + dt)

                # calculate mean squared error to judge convergence
                err = 0.0
                for j in range(state_data.size):
                    diff = state_data.flat[j] - state_prev.flat[j]
                    err += (diff.conjugate() * diff).real
                err /= state_data.size

                if err < maxerror2:
                    # fix point iteration converged
                    break
            else:
                with nb.objmode:
                    self._logger.warning(
                        "Semi-implicit Euler-Maruyama step did not converge after %d "
                        "iterations at t=%g (error=%g)",
                        maxiter,
                        t,
                        err,
                    )
                raise ConvergenceError(
                    "Semi-implicit Euler-Maruyama step did not converge."
                )
            nfev += n + 1

        self._logger.info("Init semi-implicit Euler-Maruyama stepper with dt=%g", dt)
        return implicit_step

    def _make_single_step_fixed_dt(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float], None]:
        """return a function doing a single step with an implicit Euler scheme

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the implicit step
        """
        if self.pde.is_sde:
            return self._make_single_step_fixed_dt_stochastic(state, dt)
        else:
            return self._make_single_step_fixed_dt_deterministic(state, dt)
