"""
Solvers for Poisson's and Laplace's equation

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from ..fields import ScalarField
from ..grids.base import GridBase
from ..grids.boundaries.axes import BoundariesData  # @UnusedImport
from ..tools.docstrings import fill_in_docstring


@fill_in_docstring
def solve_poisson_equation(
    rhs: ScalarField,
    bc: "BoundariesData",
    label: str = "Solution to Poisson's equation",
    **kwargs,
) -> ScalarField:
    r"""Solve Laplace's equation on a given grid

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

        where :math:`g` denotes the function specifying the outwards
        derivative for Neumann conditions. Note that for periodic boundaries
        :math:`g` vanishes, so that this condition implies that the integral
        over
        :math:`f` must vanish for neutral Neumann or periodic conditions.

    Args:
        rhs (:class:`~pde.fields.scalar.ScalarField`):
            The scalar field :math:`f` describing the right hand side
        bc:
            The boundary conditions applied to the field.
            {ARG_BOUNDARIES}
        label (str):
            The label of the returned field.

    Returns:
        :class:`~pde.fields.scalar.ScalarField`: The field :math:`u` that solves
        the equation. This field will be defined on the same grid as `rhs`.
    """
    # get the operator information
    operator = rhs.grid._get_operator_info("poisson_solver")
    # get the boundary conditions
    bcs = rhs.grid.get_boundary_conditions(bc)
    # get the actual solver
    solver = operator.factory(bcs=bcs, **kwargs)

    # solve the poisson problem
    result = ScalarField(rhs.grid, label=label)
    try:
        solver(rhs.data, result.data)
    except RuntimeError:
        magnitude = rhs.magnitude
        if magnitude > 1e-10:
            raise RuntimeError(
                "Could not solve the Poisson problem. One possible reason for this is "
                "that only periodic or Neumann conditions are applied although the "
                f"magnitude of the field is {magnitude} and thus non-zero."
            )
        else:
            raise  # another error occurred

    return result


@fill_in_docstring
def solve_laplace_equation(
    grid: GridBase, bc: "BoundariesData", label: str = "Solution to Laplace's equation"
) -> ScalarField:
    """Solve Laplace's equation on a given grid.

    This is implemented by calling :func:`solve_poisson_equation` with a
    vanishing right hand side.

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid on which the equation is solved
        bc:
            The boundary conditions applied to the field.
            {ARG_BOUNDARIES}
        label (str):
            The label of the returned field.

    Returns:
        :class:`~pde.fields.scalar.ScalarField`: The field that solves the
        equation. This field will be defined on the given `grid`.
    """
    rhs = ScalarField(grid, data=0)
    return solve_poisson_equation(rhs, bc=bc, label=label)
