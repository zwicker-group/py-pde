"""A simple diffusion equation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Callable

import numba as nb
import numpy as np

from ..fields import ScalarField
from ..grids.boundaries import set_default_bc
from ..grids.boundaries.axes import BoundariesData
from ..tools.docstrings import fill_in_docstring
from ..tools.numba import jit
from .base import PDEBase, expr_prod


class DiffusionPDE(PDEBase):
    r"""A simple diffusion equation.

    The mathematical definition is

    .. math::
        \partial_t c = D \nabla^2 c

    where :math:`c` is a scalar field and :math:`D` denotes the diffusivity.
    """

    explicit_time_dependence = False
    default_bc = "auto_periodic_neumann"
    """Default boundary condition used when no specific conditions are chosen."""

    @fill_in_docstring
    def __init__(
        self,
        diffusivity: float = 1,
        *,
        bc: BoundariesData | None = None,
        noise: float = 0,
        rng: np.random.Generator | None = None,
    ):
        """
        Args:
            diffusivity (float):
                The diffusivity of the described species
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            noise (float):
                Variance of the (additive) noise term
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
                used for stochastic simulations. Note that this random number generator
                is only used for numpy function, while compiled numba code uses the
                random number generator of numba. Moreover, in simulations using
                multiprocessing, setting the same generator in all processes might yield
                unintended correlations in the simulation results.
        """
        super().__init__(noise=noise, rng=rng)

        self.diffusivity = diffusivity
        self.bc = set_default_bc(bc, self.default_bc)

    @property
    def expression(self) -> str:
        """str: the expression of the right hand side of this PDE"""
        return expr_prod(self.diffusivity, "∇²(c)")

    def evolution_rate(  # type: ignore
        self,
        state: ScalarField,
        t: float = 0,
    ) -> ScalarField:
        """Evaluate the right hand side of the PDE.

        Args:
            state (:class:`~pde.fields.ScalarField`):
                The scalar field describing the concentration distribution
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.ScalarField`:
            Scalar field describing the evolution rate of the PDE
        """
        if not isinstance(state, ScalarField):
            raise ValueError("`state` must be ScalarField")
        laplace = state.laplace(bc=self.bc, label="evolution rate", args={"t": t})
        return self.diffusivity * laplace  # type: ignore

    def _make_pde_rhs_numba(  # type: ignore
        self, state: ScalarField
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """Create a compiled function evaluating the right hand side of the PDE.

        Args:
            state (:class:`~pde.fields.ScalarField`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`~numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`~numpy.ndarray` giving
            the evolution rate.
        """
        arr_type = nb.typeof(state.data)
        signature = arr_type(arr_type, nb.double)

        diffusivity_value = self.diffusivity
        laplace = state.grid.make_operator("laplace", bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """Compiled helper function evaluating right hand side."""
            return diffusivity_value * laplace(state_data, args={"t": t})

        return pde_rhs  # type: ignore
