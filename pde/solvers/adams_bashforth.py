"""Defines an explicit Adams-Bashforth solver.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import SolverBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..tools.typing import NumericArray, TField


class AdamsBashforthSolver(SolverBase):
    """Explicit Adams-Bashforth multi-step solver."""

    name = "adams–bashforth"

    def _make_fixed_stepper(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float, int, Any], float]:
        """Return a stepper function using an explicit scheme with fixed time steps.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            dt (float):
                Time step of the explicit stepping
        """
        if self.pde.is_sde:
            msg = "Deterministic Adams-Bashforth does not support stochastic equations"
            raise RuntimeError(msg)

        rhs_pde = self._make_pde_rhs(state)
        post_step_hook = self._make_post_step_hook(state)

        def single_step(
            state_data: NumericArray, t: float, state_prev: NumericArray
        ) -> None:
            """Perform a single Adams-Bashforth step."""
            rhs_prev = rhs_pde(state_prev, t - dt).copy()
            rhs_cur = rhs_pde(state_data, t)
            state_prev[:] = state_data  # save the previous state
            state_data += dt * (1.5 * rhs_cur - 0.5 * rhs_prev)

        # allocate memory to store the state of the previous time step
        state_prev = np.empty_like(state.data)
        init_state_prev = True

        def fixed_stepper(
            state_data: NumericArray, t_start: float, steps: int, post_step_data
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
