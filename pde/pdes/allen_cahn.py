"""A Allen-Cahn equation.

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


class AllenCahnPDE(PDEBase):
    r"""A simple Allen-Cahn equation.

    The mathematical definition is

    .. math::
        \partial_t c = \gamma \nabla^2 c - c^3 + c

    where :math:`c` is a scalar field and :math:`\gamma` sets the (squared) interfacial
    width.
    """

    explicit_time_dependence = False
    default_bc = "auto_periodic_neumann"
    """Default boundary condition used when no specific conditions are chosen."""

    interface_width: float

    @fill_in_docstring
    def __init__(
        self,
        interface_width: float = 1,
        mobility: float = 1,
        *,
        bc: BoundariesData | None = None,
    ):
        """
        Args:
            interface_width (float):
                The diffusivity of the described species
            mobility (float):
                The rate at which the structures evolve
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
        """
        super().__init__()

        self.interface_width = interface_width
        self.mobility = mobility
        self.bc = set_default_bc(bc, self.default_bc)

    @property
    def expression(self) -> str:
        """str: the expression of the right hand side of this PDE"""
        expr = f"{expr_prod(self.interface_width, '∇²c')} - c³ + c"
        if np.isclose(self.mobility, 1):
            return expr
        else:
            return expr_prod(self.mobility, f"({expr})")

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
        rhs = self.interface_width * laplace - state**3 + state
        return self.mobility * rhs  # type: ignore

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

        interface_width = self.interface_width
        mobility = self.mobility
        laplace = state.grid.make_operator("laplace", bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float) -> np.ndarray:
            """Compiled helper function evaluating right hand side."""
            return mobility * (  # type: ignore
                interface_width * laplace(state_data, args={"t": t})
                - state_data**3
                + state_data
            )

        return pde_rhs  # type: ignore
