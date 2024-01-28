"""
Defines a Crank-Nicolson solver

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


class CrankNicolsonSolver(SolverBase):
    """solving partial differential equations using the Crank-Nicolson scheme"""

    name = "crank-nicolson"

    def __init__(
        self,
        pde: PDEBase,
        *,
        maxiter: int = 100,
        maxerror: float = 1e-4,
        explicit_fraction: float = 0,
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
            explicit_fraction (float):
                Hyperparameter determinig the fraction of explicit time stepping in the
                implicit step. `explicit_fraction == 0` is the simple Crank-Nicolson
                scheme, while `explicit_fraction == 1` reduces to the explicit Euler
                method. Intermediate values can improve convergence.
            backend (str):
                Determines how the function is created. Accepted  values are 'numpy` and
                'numba'. Alternatively, 'auto' lets the code decide for the most optimal
                backend.
        """
        super().__init__(pde, backend=backend)
        self.maxiter = maxiter
        self.maxerror = maxerror
        self.explicit_fraction = explicit_fraction

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
            raise RuntimeError("Cannot use implicit stepper with stochastic equation")

        self.info["function_evaluations"] = 0
        self.info["scheme"] = "implicit-euler"
        self.info["stochastic"] = False
        self.info["dt_adaptive"] = False

        rhs = self._make_pde_rhs(state, backend=self.backend)
        maxiter = int(self.maxiter)
        maxerror2 = self.maxerror**2
        α = self.explicit_fraction

        # handle deterministic version of the pde
        def crank_nicolson_step(state_data: np.ndarray, t: float) -> None:
            """compiled inner loop for speed"""
            nfev = 0  # count function evaluations

            # keep values at the current time t point used in iteration
            state_t = state_data.copy()
            rate_t = rhs(state_t, t)

            # new state is weighted average of explicit and Crank-Nicolson steps
            state_cn = state_t + dt / 2 * (rhs(state_data, t + dt) + rate_t)
            state_data[:] = α * state_data + (1 - α) * state_cn
            state_prev = np.empty_like(state_data)

            # fixed point iteration for improving state after dt
            for n in range(maxiter):
                state_prev[:] = state_data  # keep previous state to judge convergence
                # new state is weighted average of explicit and Crank-Nicolson steps
                state_cn = state_t + dt / 2 * (rhs(state_data, t + dt) + rate_t)
                state_data[:] = α * state_data + (1 - α) * state_cn

                # calculate mean squared error
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
                        "Crank-Nicolson step did not converge after %d iterations "
                        "at t=%g (error=%g)",
                        maxiter,
                        t,
                        err,
                    )
                raise ConvergenceError("Crank-Nicolson step did not converge.")
            nfev += n + 2

        return crank_nicolson_step
