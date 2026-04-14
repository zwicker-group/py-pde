"""A Cahn-Hilliard equation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..fields import ScalarField
from ..grids.boundaries import set_default_bc
from ..tools.docstrings import fill_in_docstring
from .base import PDEBase, expr_prod

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..backends import BackendBase
    from ..grids.boundaries.axes import BoundariesData
    from ..tools.typing import TNativeArray


class CahnHilliardPDE(PDEBase):
    r"""A simple Cahn-Hilliard equation.

    The mathematical definition is

    .. math::
        \partial_t c = \nabla^2 \left(c^3 - c - \gamma \nabla^2 c\right)

    where :math:`c` is a scalar field taking values on the interval :math:`[-1, 1]` and
    :math:`\gamma` sets the (squared) interfacial width.
    """

    explicit_time_dependence = False
    default_bc_c = "auto_periodic_neumann"
    """Default boundary condition for order parameter."""
    default_bc_mu = "auto_periodic_neumann"
    """Default boundary condition for chemical potential."""

    @fill_in_docstring
    def __init__(
        self,
        interface_width: float = 1,
        *,
        bc_c: BoundariesData | None = None,
        bc_mu: BoundariesData | None = None,
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
        self.bc_c: BoundariesData = set_default_bc(bc_c, self.default_bc_c)
        self.bc_mu: BoundariesData = set_default_bc(bc_mu, self.default_bc_mu)

    @property
    def expression(self) -> str:
        """str: the expression of the right hand side of this PDE"""
        return f"∇²(c³ - c - {expr_prod(self.interface_width, '∇²c')})"

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
        assert isinstance(state, ScalarField), "`state` must be ScalarField"
        c_laplace = state.laplace(bc=self.bc_c, label="evolution rate", args={"t": t})
        result = state**3 - state - self.interface_width * c_laplace
        return result.laplace(bc=self.bc_mu, args={"t": t})  # type: ignore

    def make_evolution_rate(
        self, state: ScalarField, backend: BackendBase
    ) -> Callable[[TNativeArray, float], TNativeArray]:
        """Create a compiled function evaluating the right hand side of the PDE.

        Args:
            state (:class:`~pde.fields.ScalarField`):
                An example for the state defining the grid and data types
            backend (str or :class:`~pde.backends.base.BackendBase`):
                The backend used for numerical operations

        Returns:
            A function with signature `(state_data, t)`, which can be called with an
            instance of the state data and time to obtain the associated evolution rate.
        """
        interface_width = self.interface_width
        args: dict[str, Any] = {"backend": backend, "dtype": state.dtype}
        laplace_c = state.grid.make_operator(operator="laplace", bc=self.bc_c, **args)
        laplace_mu = state.grid.make_operator(operator="laplace", bc=self.bc_mu, **args)

        def pde_rhs(state_data, t=0):
            """Evaluate right hand side of PDE."""
            mu = (
                state_data**3
                - state_data
                - interface_width * laplace_c(state_data, args={"t": t})
            )
            return laplace_mu(mu, args={"t": t})

        return pde_rhs
