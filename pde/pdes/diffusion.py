"""
A simple diffusion equation

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


class DiffusionPDE(PDEBase):
    r"""A simple diffusion equation

    The mathematical definition is

    .. math::
        \partial_t c = D \nabla^2 c

    where :math:`c` is a scalar field and :math:`D` denotes the diffusivity.
    """

    explicit_time_dependence = False

    @fill_in_docstring
    def __init__(
        self,
        diffusivity: float = 1,
        noise: float = 0,
        bc: BoundariesData = "auto_periodic_neumann",
    ):
        """
        Args:
            diffusivity (float):
                The diffusivity of the described species
            noise (float):
                Variance of the (additive) noise term
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
        """
        super().__init__(noise=noise)

        self.diffusivity = diffusivity
        self.bc = bc

    @property
    def expression(self) -> str:
        """str: the expression of the right hand side of this PDE"""
        return expr_prod(self.diffusivity, "∇²(c)")

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
        laplace = state.laplace(bc=self.bc, label="evolution rate", args={"t": t})
        return self.diffusivity * laplace  # type: ignore

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

        diffusivity_value = self.diffusivity
        laplace = state.grid.make_operator("laplace", bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """compiled helper function evaluating right hand side"""
            return diffusivity_value * laplace(state_data, args={"t": t})

        return pde_rhs  # type: ignore
