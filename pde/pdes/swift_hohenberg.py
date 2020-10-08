"""
The Swift-Hohenberg equation

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable

import numpy as np

from ..fields import ScalarField
from ..grids.boundaries.axes import BoundariesData
from ..tools.docstrings import fill_in_docstring
from ..tools.numba import jit, nb
from .base import PDEBase, expr_prod


class SwiftHohenbergPDE(PDEBase):
    r"""The Swift-Hohenberg equation

    The mathematical definition is

    .. math::
        \partial_t c =
            \left[\epsilon - \left(k_c^2 + \nabla^2\right)^2\right] c
            + \delta \, c^2 - c^3

    where :math:`c` is a scalar field and :math:`\epsilon`, :math:`k_c^2`, and
    :math:`\delta` are parameters of the equation.
    """

    explicit_time_dependence = False

    @fill_in_docstring
    def __init__(
        self,
        rate: float = 0.1,
        kc2: float = 1.0,
        delta: float = 1.0,
        bc: BoundariesData = "natural",
        bc_lap: BoundariesData = None,
    ):
        r"""
        Args:
            rate (float):
                The bifurcation parameter :math:`\epsilon`
            kc2 (float):
                Squared wave vector :math:`k_c^2` of the linear instability
            delta (float):
                Parameter :math:`\delta` of the non-linearity
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            bc_lap:
                The boundary conditions applied to the second derivative of the
                scalar field :math:`c`. If `None`, the same boundary condition
                as `bc` is chosen. Otherwise, this supports the same options as
                `bc`.
        """
        super().__init__()

        self.rate = rate
        self.kc2 = kc2
        self.delta = delta
        self.bc = bc
        self.bc_lap = bc if bc_lap is None else bc_lap

    @property
    def expression(self) -> str:
        """ str: the expression of the right hand side of this PDE """
        return (
            f"{expr_prod(self.rate - self.kc2 ** 2, 'c')} - c**3"
            f" + {expr_prod(self.delta, 'c**2')}"
            f" - laplace({expr_prod(2 * self.kc2, 'c')} + laplace(c))"
        )

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
        assert isinstance(state, ScalarField)
        state_laplace = state.laplace(bc=self.bc)
        state_laplace2 = state_laplace.laplace(bc=self.bc_lap)

        result = (
            (self.rate - self.kc2 ** 2) * state
            - 2 * self.kc2 * state_laplace
            - state_laplace2
            + self.delta * state ** 2
            - state ** 3
        )
        result.label = "evolution rate"
        return result  # type: ignore

    def _make_pde_rhs_numba(self, state: ScalarField) -> Callable:  # type: ignore
        """create a compiled function evaluating the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.ScalarField`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`numpy.ndarray` giving
            the evolution rate.
        """
        shape = state.grid.shape
        arr_type = nb.typeof(np.empty(shape, dtype=state.data.dtype))
        signature = arr_type(arr_type, nb.double)

        rate = self.rate
        kc2 = self.kc2
        delta = self.delta

        laplace = state.grid.get_operator("laplace", bc=self.bc)
        laplace2 = state.grid.get_operator("laplace", bc=self.bc_lap)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """ compiled helper function evaluating right hand side """
            state_laplace = laplace(state_data)
            state_laplace2 = laplace2(state_laplace)

            return (
                (rate - kc2 ** 2) * state_data
                - 2 * kc2 * state_laplace
                - state_laplace2
                + delta * state_data ** 2
                - state_data ** 3
            )

        return pde_rhs  # type: ignore
