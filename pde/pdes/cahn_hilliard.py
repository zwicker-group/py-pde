"""
A Cahn-Hilliard equation

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable

import numba as nb
import numpy as np

from ..fields import ScalarField
from ..grids.boundaries.axes import BoundariesData
from ..tools.docstrings import fill_in_docstring
from ..tools.numba import jit
from .base import PDEBase, expr_prod


class CahnHilliardPDE(PDEBase):
    r"""A simple Cahn-Hilliard equation

    The mathematical definition is

    .. math::
        \partial_t c = \nabla^2 \left(c^3 - c - \gamma \nabla^2 c\right)

    where :math:`c` is a scalar field taking values on the interval :math:`[-1, 1]` and
    :math:`\gamma` sets the (squared) interfacial width.
    """

    explicit_time_dependence = False

    @fill_in_docstring
    def __init__(
        self,
        interface_width: float = 1,
        bc_c: BoundariesData = "auto_periodic_neumann",
        bc_mu: BoundariesData = "auto_periodic_neumann",
    ):
        """
        Args:
            interface_width (float):
                The width of the interface between the separated phases. This defines
                a characteristic length in the simulation. The grid needs to resolve
                this length of a stable simulation.
            bc_c:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            bc_mu:
                The boundary conditions applied to the chemical potential associated
                with the scalar field :math:`c`. Supports the same options as `bc_c`.
        """
        super().__init__()

        self.interface_width = interface_width
        self.bc_c = bc_c
        self.bc_mu = bc_mu

    @property
    def expression(self) -> str:
        """str: the expression of the right hand side of this PDE"""
        return f"∇²(c³ - c - {expr_prod(self.interface_width, '∇²c')})"

    def evolution_rate(  # type: ignore
        self,
        state: ScalarField,
        t: float = 0,
    ) -> ScalarField:
        """evaluate the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.ScalarField`):
                The scalar field describing the concentration distribution
            t (float): The current time point

        Returns:
            :class:`~pde.fields.ScalarField`:
            Scalar field describing the evolution rate of the PDE
        """
        assert isinstance(state, ScalarField), "`state` must be ScalarField"
        c_laplace = state.laplace(bc=self.bc_c, label="evolution rate", args={"t": t})
        result = state**3 - state - self.interface_width * c_laplace
        return result.laplace(bc=self.bc_mu, args={"t": t})  # type: ignore

    def _make_pde_rhs_numba(  # type: ignore
        self, state: ScalarField
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """create a compiled function evaluating the right hand side of the PDE

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
        laplace_c = state.grid.make_operator("laplace", bc=self.bc_c)
        laplace_mu = state.grid.make_operator("laplace", bc=self.bc_mu)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """compiled helper function evaluating right hand side"""
            mu = (
                state_data**3
                - state_data
                - interface_width * laplace_c(state_data, args={"t": t})
            )
            return laplace_mu(mu, args={"t": t})

        return pde_rhs  # type: ignore
