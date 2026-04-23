"""Defines an explicit Milstein solver for stochastic differential equations.

.. autosummary::
   :nosignatures:

   MilsteinSolver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..tools.misc import get_array_namespace
from .euler import _DUMMY_FUNCTION_2ARGS, EulerSolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from pde.tools.typing import NumericArray, TField

    from ..backends.base import BackendBase
    from ..pdes.base import PDEBase


class MilsteinSolver(EulerSolver):
    """Milstein method for stochastic differential equations."""

    name = "milstein"

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
        if not pde.use_noise_variance:
            msg = "Milstein solver requires `use_noise_variance` enabled."
            raise RuntimeError(msg)

    def _make_single_step_fixed_dt_stochastic(
        self, state: TField, dt: float
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Make a Euler-Milstein single-step update with fixed time step.

        Info:
            This solver should only be used for problems with multiplicative noise.
            While the solver works for additive noise, the extra corrections might slow
            down calculations.

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
        assert self.pde.use_noise_variance
        noise_drift_factor = self._noise_drift_factor
        fn = self.pde.make_noise_variance(  # type: ignore
            state, backend=self.backend, ret_diff=True
        )
        noise_var = self.backend.compile_function(fn)
        gaussian_noise = self.backend.make_gaussian_noise(state, rng=self.pde.rng)

        # handle with second noise interface based on supplying a realization
        if use_noise_realization := self.pde.use_noise_realization:
            rhs_noise = self.pde._make_noise_realization(state, backend=self.backend)  # type: ignore
        else:
            rhs_noise = _DUMMY_FUNCTION_2ARGS
        rhs_noise = self.backend.compile_function(rhs_noise)

        # noise increment scales with square root of time step
        dt_sqrt = np.sqrt(dt)

        def single_step(state_data: NumericArray, t: float) -> NumericArray:
            """Perform a single Euler-Milstein step."""
            # support any backend following Python Array API
            nx = get_array_namespace(state_data)

            # evaluate deterministic part and variance without modifying field, yet
            evolution_rate = rhs_pde(state_data, t)
            noise_var_field, noise_var_diff_field = noise_var(state_data, t)

            # handle second noise interface
            if use_noise_realization:
                noise_realization = rhs_noise(state_data, t)
                if noise_realization is not None:
                    state_data += dt_sqrt * noise_realization

            # apply the deterministic part and the additive noise
            dW = dt_sqrt * gaussian_noise()
            state_data += (
                dt * evolution_rate
                + 0.5 * dt * noise_drift_factor * noise_var_diff_field
                + nx.sqrt(noise_var_field) * dW
                + 0.25 * noise_var_diff_field * (dW**2 - dt)
            )

            return state_data

        self._logger.info(
            "Initialize explicit Euler-Milstein single-step update with dt=%g", dt
        )
        return single_step
