"""
The Kardar–Parisi–Zhang (KPZ) equation describing the evolution of an interface

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


class KPZInterfacePDE(PDEBase):
    r"""The Kardar–Parisi–Zhang (KPZ) equation

    The mathematical definition is

    .. math::
        \partial_t h = \nu \nabla^2 h +
            \frac{\lambda}{2} \left(\nabla h\right)^2  + \eta(\boldsymbol r, t)

    where :math:`h` is the height of the interface in Monge parameterization. The
    dynamics are governed by the two parameters :math:`\nu` and :math:`\lambda`, while
    :math:`\eta` is Gaussian white noise, whose strength is controlled by the `noise`
    argument.
    """

    explicit_time_dependence = False

    @fill_in_docstring
    def __init__(
        self,
        nu: float = 0.5,
        lmbda: float = 1,
        *,
        noise: float = 0,
        bc: BoundariesData = "auto_periodic_neumann",
    ):
        r"""
        Args:
            nu (float):
                Parameter :math:`\nu` for the strength of the diffusive term
            lmbda (float):
                Parameter :math:`\lambda` for the strenth of the gradient term
            noise (float):
                Strength of the (additive) noise term
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
        """
        super().__init__(noise=noise)

        self.nu = nu
        self.lmbda = lmbda
        self.bc = bc

    @property
    def expression(self) -> str:
        """str: the expression of the right hand side of this PDE"""
        return expr_prod(self.nu, "∇²c") + " + " + expr_prod(self.lmbda, "|∇c|²")

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
        result = self.nu * state.laplace(bc=self.bc, args={"t": t})
        result += self.lmbda * state.gradient_squared(bc=self.bc, args={"t": t})
        result.label = "evolution rate"
        return result  # type: ignore

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

        nu_value, lambda_value = self.nu, self.lmbda
        laplace = state.grid.make_operator("laplace", bc=self.bc)
        gradient_squared = state.grid.make_operator("gradient_squared", bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """compiled helper function evaluating right hand side"""
            result = nu_value * laplace(state_data, args={"t": t})
            result += lambda_value * gradient_squared(state_data, args={"t": t})
            return result

        return pde_rhs  # type: ignore
