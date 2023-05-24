"""
Defines an implicit solver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable

import numba as nb
import numpy as np

from ..fields.base import FieldBase
from ..pdes.base import PDEBase
from .base import SolverBase


class ConvergenceError(RuntimeError):
    pass


class ImplicitSolver(SolverBase):
    """class for solving partial differential equations implicitly"""

    name = "implicit"

    def __init__(
        self,
        pde: PDEBase,
        maxiter: int = 100,
        maxerror: float = 1e-4,
        backend: str = "auto",
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

        # handle deterministic version of the pde
        def implicit_step(state_data: np.ndarray, t: float) -> None:
            """compiled inner loop for speed"""
            nfev = 0  # count function evaluations

            # estimate state at next time point
            evolution_last = dt * rhs(state_data, t)

            for n in range(maxiter):
                # fixed point iteration for improving state after dt
                state_guess = state_data + evolution_last
                evolution_this = dt * rhs(state_guess, t + dt)

                # calculate mean squared error
                err = 0.0
                for j in range(state_data.size):
                    diff = (
                        state_guess.flat[j]
                        - state_data.flat[j]
                        - evolution_this.flat[j]
                    )
                    err += (diff.conjugate() * diff).real
                err /= state_data.size

                if err < maxerror2:
                    # fix point iteration converged
                    break

                evolution_last = evolution_this
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
            state_data += evolution_this

        return implicit_step
