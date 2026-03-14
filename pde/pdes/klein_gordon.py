"""The Klein-Gordon equation.

.. codeauthor:: Greg Partin <gpartin@gmail.com>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..fields import FieldCollection, ScalarField
from ..grids.boundaries import set_default_bc
from ..tools.docstrings import fill_in_docstring
from .base import PDEBase, expr_prod

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..backends import BackendBase
    from ..grids.boundaries.axes import BoundariesData
    from ..tools.typing import TArray


class KleinGordonPDE(PDEBase):
    r"""The Klein-Gordon equation.

    The mathematical definition,
    :math:`\partial_t^2 u = c^2 \nabla^2 u - \mu^2 u`,
    is implemented as two first-order equations,

    .. math::
        \partial_t u &= v \\
        \partial_t v &= c^2 \nabla^2 u - \mu^2 u

    where :math:`c` sets the wave speed, :math:`\mu` is the mass parameter, and
    :math:`v` is an auxiliary field. Note that the class expects an initial condition
    specifying both fields, which can be created using the
    :meth:`KleinGordonPDE.get_initial_condition` method. The result will also return
    two fields.

    The Klein-Gordon equation describes relativistic scalar fields and reduces to the
    standard wave equation when :math:`\mu = 0`.
    """

    explicit_time_dependence = False
    default_bc = "auto_periodic_neumann"
    """Default boundary condition used when no specific conditions are chosen."""

    @fill_in_docstring
    def __init__(
        self,
        speed: float = 1,
        mass: float = 1,
        *,
        bc: BoundariesData | None = None,
    ):
        """
        Args:
            speed (float):
                The speed :math:`c` of the wave
            mass (float):
                The mass parameter :math:`\mu`
            bc:
                The boundary conditions applied to the field :math:`u`.
                {ARG_BOUNDARIES}
        """
        super().__init__()

        self.speed = speed
        self.mass = mass
        self.bc = set_default_bc(bc, self.default_bc)

    def get_initial_condition(self, u: ScalarField, v: ScalarField | None = None):
        """Create a suitable initial condition.

        Args:
            u (:class:`~pde.fields.ScalarField`):
                The initial field amplitude on the grid
            v (:class:`~pde.fields.ScalarField`, optional):
                The initial rate of change. This is assumed to be zero if the
                value is omitted.

        Returns:
            :class:`~pde.fields.FieldCollection`:
                The combined fields u and v, suitable for the simulation
        """
        if v is None:
            v = ScalarField(u.grid)
        return FieldCollection([u, v], labels=["u", "v"])

    @property
    def expressions(self) -> dict[str, str]:
        """dict: the expressions of the right hand side of this PDE"""
        v_expr = expr_prod(self.speed**2, "∇²u")
        mass_expr = expr_prod(self.mass**2, "u")
        if self.mass == 0:
            return {"u": "v", "v": v_expr}
        return {"u": "v", "v": f"{v_expr} - {mass_expr}"}

    def evolution_rate(  # type: ignore
        self,
        state: FieldCollection,
        t: float = 0,
    ) -> FieldCollection:
        """Evaluate the right hand side of the PDE.

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields :math:`u` and :math:`v`
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.FieldCollection`:
            Fields describing the evolution rates of the PDE
        """
        if not isinstance(state, FieldCollection):
            msg = "`state` must be FieldCollection"
            raise TypeError(msg)
        if len(state) != 2:
            msg = "`state` must contain two fields"
            raise ValueError(msg)
        u, v = state
        u_t = v.copy()
        v_t = self.speed**2 * u.laplace(self.bc, args={"t": t}) - self.mass**2 * u
        return FieldCollection([u_t, v_t])

    def make_evolution_rate(
        self, state: FieldCollection, backend: BackendBase
    ) -> Callable[[TArray, float], TArray]:
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
        speed2 = self.speed**2
        mass2 = self.mass**2
        laplace = state.grid.make_operator(
            operator="laplace",
            bc=self.bc,
            backend=backend,
            native=True,
            dtype=state.dtype,
        )

        def pde_rhs(state_data, t=0):
            """Evaluate right hand side of PDE."""
            return np.stack(
                (
                    state_data[1],
                    speed2 * laplace(state_data[0], args={"t": t})
                    - mass2 * state_data[0],
                )
            )

        return pde_rhs
