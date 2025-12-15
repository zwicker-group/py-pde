"""Defines an implicit Euler solver.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .base import ConvergenceError, SolverBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..pdes.base import PDEBase
    from ..tools.typing import BackendType, NumericArray, TField


class ImplicitSolver(SolverBase):
    """Implicit (backward) Euler PDE solver."""

    name = "implicit"

    def __init__(
        self,
        pde: PDEBase,
        *,
        maxiter: int = 100,
        maxerror: float = 1e-4,
        backend: BackendType | Literal["auto"] = "auto",
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
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
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], None]:
        """Return a function doing a deterministic step with an implicit Euler scheme.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the implicit step
        """
        if self.pde.is_sde:
            msg = "Deterministic implicit Euler does not support stochastic equations"
            raise RuntimeError(msg)

        self.info["function_evaluations"] = 0
        self.info["scheme"] = "implicit-euler"
        self.info["stochastic"] = False

        rhs = self._make_pde_rhs(state)
        maxiter = int(self.maxiter)
        maxerror2 = self.maxerror**2

        # handle deterministic version of the pde
        def implicit_step(state_data: NumericArray, t: float) -> None:
            """Compiled inner loop for speed."""
            nfev = 0  # count function evaluations

            # save state at current time point t for stepping
            state_t = state_data.copy()

            # estimate state at next time point
            state_data[:] = state_t + dt * rhs(state_data, t)
            state_prev = np.empty_like(state_data)

            # fixed point iteration for improving state after dt
            for n in range(maxiter):
                state_prev[:] = state_data  # keep previous state to judge convergence
                # another iteration to improve estimate
                state_data[:] = state_t + dt * rhs(state_data, t + dt)

                # calculate mean squared error to judge convergence
                err = 0.0
                for j in range(state_data.size):
                    diff: NumericArray = state_data.flat[j] - state_prev.flat[j]
                    err += (diff.conjugate() * diff).real
                err /= state_data.size

                if err < maxerror2:
                    # fix point iteration converged
                    break
            else:
                msg = "Implicit Euler step did not converge."
                raise ConvergenceError(msg)
            nfev += n + 1

        self._logger.info("Init implicit Euler stepper with dt=%g", dt)
        return implicit_step

    def _make_single_step_fixed_dt_stochastic(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], None]:
        """Return a function doing a step for a SDE with an implicit Euler scheme.

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

        rhs = self._make_pde_rhs(state)
        rhs_noise = self.pde.make_noise_realization(state, backend=self.backend)  # type: ignore
        maxiter = int(self.maxiter)
        maxerror2 = self.maxerror**2

        # handle deterministic version of the pde
        def implicit_step(state_data: NumericArray, t: float) -> None:
            """Compiled inner loop for speed."""
            nfev = 0  # count function evaluations

            # save state at current time point t for stepping
            state_t = state_data.copy()
            state_prev = np.empty_like(state_data)

            # estimate state at next time point
            evolution_rate = rhs(state_data, t)
            noise_realization = rhs_noise(state_data, t)
            if noise_realization is not None:
                # add the noise to the reference state at the current time point and
                # adept the state at the next time point iteratively below
                state_t += np.sqrt(dt) * noise_realization
            state_data[:] = state_t + dt * evolution_rate  # estimated new state

            # fixed point iteration for improving state after dt
            for n in range(maxiter):
                state_prev[:] = state_data  # keep previous state to judge convergence
                # another iteration to improve estimate
                state_data[:] = state_t + dt * rhs(state_data, t + dt)

                # calculate mean squared error to judge convergence
                err = 0.0
                for j in range(state_data.size):
                    diff: NumericArray = state_data.flat[j] - state_prev.flat[j]
                    err += (diff.conjugate() * diff).real
                err /= state_data.size

                if err < maxerror2:
                    # fix point iteration converged
                    break
            else:
                msg = "Semi-implicit Euler-Maruyama step did not converge."
                raise ConvergenceError(msg)
            nfev += n + 1

        self._logger.info("Init semi-implicit Euler-Maruyama stepper with dt=%g", dt)
        return implicit_step

    def _make_single_step_fixed_dt(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], None]:
        """Return a function doing a single step with an implicit Euler scheme.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the implicit step
        """
        if self.pde.is_sde:
            return self._make_single_step_fixed_dt_stochastic(state, dt)
        return self._make_single_step_fixed_dt_deterministic(state, dt)
