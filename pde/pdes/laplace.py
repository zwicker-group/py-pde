"""Solvers for Poisson's and Laplace's equation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backends import backends
from ..fields import ScalarField, VectorField
from ..tools.docstrings import fill_in_docstring

if TYPE_CHECKING:
    from ..grids.base import GridBase
    from ..grids.boundaries.axes import BoundariesData


@fill_in_docstring
def solve_poisson_equation(
    rhs: ScalarField,
    bc: BoundariesData,
    *,
    backend: str = "scipy",
    label: str = "Solution to Poisson's equation",
    **kwargs,
) -> ScalarField:
    r"""Solve Laplace's equation on a given grid.

    Denoting the current field by :math:`u`, we thus solve for :math:`f`, defined by the
    equation

    .. math::
        \nabla^2 u(\boldsymbol r) = -f(\boldsymbol r)

    with boundary conditions specified by `bc`.

    Note:
        In case of periodic or Neumann boundary conditions, the right hand side
        :math:`f(\boldsymbol r)` needs to satisfy the following condition

        .. math::
            \int f \, \mathrm{d}V = \oint g \, \mathrm{d}S \;,

        where :math:`g` denotes the function specifying the outwards derivative for
        Neumann conditions. Note that for periodic boundaries :math:`g` vanishes, so
        that this condition implies that the integral over :math:`f` must vanish for
        neutral Neumann or periodic conditions.

    Args:
        rhs (:class:`~pde.fields.scalar.ScalarField`):
            The scalar field :math:`f` describing the right hand side
        bc:
            The boundary conditions applied to the field.
            {ARG_BOUNDARIES}
        backend (str):
            The name of the backend to use to implement this operator.
        label (str):
            The label of the returned field.
        **kwargs:
            Additional parameters influence how the Laplace operator is constructed.

    Returns:
        :class:`~pde.fields.scalar.ScalarField`: Field :math:`u` solving the equation.
    """
    # get the operator information
    operator = backends["scipy"].get_operator_info(rhs.grid, "poisson_solver")
    # get the boundary conditions
    bcs = rhs.grid.get_boundary_conditions(bc)
    # get the actual solver
    solver = operator.factory(bcs=bcs, **kwargs)

    # solve the poisson problem
    result = ScalarField(rhs.grid, label=label)
    try:
        solver(rhs.data, result.data)
    except RuntimeError as err:
        magnitude = rhs.magnitude
        if magnitude > 1e-10:
            msg = (
                "Could not solve the Poisson problem. One possible reason for this is "
                "that only periodic or Neumann conditions are applied although the "
                f"magnitude of the field is {magnitude} and thus non-zero."
            )
            raise RuntimeError(msg) from err
        raise  # another error occurred

    return result


@fill_in_docstring
def solve_laplace_equation(
    grid: GridBase,
    bc: BoundariesData,
    *,
    backend: str = "scipy",
    label: str = "Solution to Laplace's equation",
) -> ScalarField:
    """Solve Laplace's equation on a given grid.

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid on which the equation is solved
        bc:
            The boundary conditions applied to the field.
            {ARG_BOUNDARIES}
        backend (str):
            The name of the backend to use to implement this operator.
        label (str):
            The label of the returned field.

    Returns:
        :class:`~pde.fields.scalar.ScalarField`: The field that solves the equation.
    """
    rhs = ScalarField(grid, data=0)
    return solve_poisson_equation(rhs, bc=bc, label=label)


@fill_in_docstring
def helmholtz_decomposition(
    field: VectorField, bc: BoundariesData
) -> tuple[ScalarField, VectorField]:
    r"""Return Helmholtz decomposition of a vector field.

    For a vector field :math:`\boldsymbol u`, we return a scalar potential :math:`\phi`
    and a solenoidal (divergence-free) vector field :math:`\boldsymbol v`, which obey

    .. math::
        \boldsymbol u = \nabla \phi + \boldsymbol v


    Args:
        field (:class:`~pde.fields.vectorial.VectorField`):
            The vector field :math:`u` that needs to be decomposed.
        bc:
            The boundary conditions applied to the field. Note that the same boundary
            conditions are also applied to when solving the Poisson equation to
            determine the potential.
            {ARG_BOUNDARIES}

    Returns:
        :class:`~pde.fields.ScalarField`, :class:`~pde.fields.VectorField`:
            The two fields of the Helmholtz decomposition
    """
    bcs = field.grid.get_boundary_conditions(bc)
    source = field.divergence(bcs)
    potential = solve_poisson_equation(source, bcs)
    solenoidal: VectorField = field - potential.gradient(bcs)  # type: ignore
    return potential, solenoidal
