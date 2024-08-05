"""Defines an explicit Adams-Bashforth solver.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Any, Callable

import numba as nb
import numpy as np

from ..fields.base import FieldBase
from ..tools.numba import jit
from .base import SolverBase


class AdamsBashforthSolver(SolverBase):
    """Explicit Adams-Bashforth multi-step solver."""

    name = "adams–bashforth"

    def _make_fixed_stepper(
        self, state: FieldBase, dt: float
    ) -> Callable[[np.ndarray, float, int, Any], float]:
        """Return a stepper function using an explicit scheme with fixed time steps.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping
        """
        if self.pde.is_sde:
            raise NotImplementedError

        rhs_pde = self._make_pde_rhs(state, backend=self.backend)
        post_step_hook = self._make_post_step_hook(state)

        def single_step(
            state_data: np.ndarray, t: float, state_prev: np.ndarray
        ) -> None:
            """Perform a single Adams-Bashforth step."""
            rhs_prev = rhs_pde(state_prev, t - dt).copy()
            rhs_cur = rhs_pde(state_data, t)
            state_prev[:] = state_data  # save the previous state
            state_data += dt * (1.5 * rhs_cur - 0.5 * rhs_prev)

        # allocate memory to store the state of the previous time step
        state_prev = np.empty_like(state.data)
        init_state_prev = True

        if self._compiled:
            sig_single_step = (nb.typeof(state.data), nb.double, nb.typeof(state_prev))
            single_step = jit(sig_single_step)(single_step)

        def fixed_stepper(
            state_data: np.ndarray, t_start: float, steps: int, post_step_data
        ) -> float:
            """Perform `steps` steps with fixed time steps."""
            nonlocal state_prev, init_state_prev

            if init_state_prev:
                # initialize the state_prev with an estimate of the previous step
                state_prev[:] = state_data - dt * rhs_pde(state_data, t_start)
                init_state_prev = False

            for i in range(steps):
                # calculate the right hand side
                t = t_start + i * dt
                single_step(state_data, t, state_prev)
                post_step_hook(state_data, t, post_step_data=post_step_data)

            return t + dt

        self._logger.info("Init explicit Adams-Bashforth stepper with dt=%g", dt)

        return fixed_stepper
