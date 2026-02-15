"""Defines a Crank-Nicolson solver.

.. autosummary::
   :nosignatures:

   CrankNicolsonSolver

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


class CrankNicolsonSolver(SolverBase):
    """Crank-Nicolson solver."""

    name = "crank-nicolson"

    def __init__(
        self,
        pde: PDEBase,
        *,
        maxiter: int = 100,
        maxerror: float = 1e-4,
        explicit_fraction: float = 0,
        backend: BackendType | Literal["auto"] = "auto",
    ):
        """
        Args:
            pde (:class:`~pde.pdes.base.PDEBase`):
                The partial differential equation that should be solved
            maxiter (int):
                Maximum number of iterations for the implicit solver
            maxerror (float):
                Maximum error tolerance for the implicit solver
            explicit_fraction (float):
                Fraction of explicit time stepping (0 for fully implicit)
            backend (str):
                The backend used for numerical operations
        """
        super().__init__(pde, backend=backend)
        self.maxiter = maxiter
        self.maxerror = maxerror
        self.explicit_fraction = explicit_fraction

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
            msg = "Deterministic Crank-Nicolson does not support stochastic equations"
            raise RuntimeError(msg)

        self.info["function_evaluations"] = 0
        self.info["stochastic"] = False

        rhs = self._backend_obj.make_pde_rhs(self.pde, state)
        maxiter = int(self.maxiter)
        maxerror2 = self.maxerror**2
        α = self.explicit_fraction

        # handle deterministic version of the pde
        def crank_nicolson_step(state_data: NumericArray, t: float) -> None:
            """Compiled inner loop for speed."""
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
                    diff: NumericArray = state_data.flat[j] - state_prev.flat[j]
                    err += (np.conj(diff) * diff).real  # type: ignore
                err /= state_data.size

                if err < maxerror2:
                    # fix point iteration converged
                    break
            else:
                msg = "Crank-Nicolson step did not converge."
                raise ConvergenceError(msg)
            nfev += n + 2

        return crank_nicolson_step
