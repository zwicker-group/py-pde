"""
The Kardar–Parisi–Zhang (KPZ) equation describing the evolution of an interface

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable, Optional

import numba as nb
import numpy as np

from ..fields import ScalarField
from ..grids.boundaries.axes import BoundariesData
from ..tools.docstrings import fill_in_docstring
from ..tools.numba import jit
from .base import PDEBase, expr_prod


class KuramotoSivashinskyPDE(PDEBase):
    r"""The Kuramoto-Sivashinsky equation

    The mathematical definition is

    .. math::
        \partial_t u = -\nu \nabla^4 u  - \nabla^2 u -
            \frac{1}{2} \left(\nabla h\right)^2  + \eta(\boldsymbol r, t)

    where :math:`u` is the height of the interface in Monge parameterization. The
    dynamics are governed by the parameters :math:`\nu` , while :math:`\eta` is Gaussian
    white noise, whose strength is controlled by the `noise` argument.
    """

    explicit_time_dependence = False

    @fill_in_docstring
    def __init__(
        self,
        nu: float = 1,
        *,
        noise: float = 0,
        bc: BoundariesData = "auto_periodic_neumann",
        bc_lap: Optional[BoundariesData] = None,
    ):
        r"""
        Args:
            nu (float):
                Parameter :math:`\nu` for the strength of the fourth-order term
            noise (float):
                Strength of the (additive) noise term
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            bc_lap:
                The boundary conditions applied to the second derivative of the
                scalar field :math:`c`. If `None`, the same boundary condition
                as `bc` is chosen. Otherwise, this supports the same options as
                `bc`.
        """
        super().__init__(noise=noise)

        self.nu = nu
        self.bc = bc
        self.bc_lap = bc if bc_lap is None else bc_lap

    @property
    def expression(self) -> str:
        """str: the expression of the right hand side of this PDE"""
        expr = f"c + {expr_prod(self.nu, '∇²c')}"
        return f"-∇²({expr}) - 0.5 * |∇c|²"

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
        state_lap = state.laplace(bc=self.bc, args={"t": t})
        result = (
            -self.nu * state_lap.laplace(bc=self.bc_lap, args={"t": t})
            - state_lap
            - 0.5 * state.gradient_squared(bc=self.bc, args={"t": t})
        )
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

        nu_value = self.nu
        laplace = state.grid.make_operator("laplace", bc=self.bc)
        laplace2 = state.grid.make_operator("laplace", bc=self.bc_lap)
        gradient_sq = state.grid.make_operator("gradient_squared", bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """compiled helper function evaluating right hand side"""
            result = -laplace(state_data, args={"t": t})
            result += nu_value * laplace2(result, args={"t": t})
            result -= 0.5 * gradient_sq(state_data, args={"t": t})
            return result

        return pde_rhs  # type: ignore
