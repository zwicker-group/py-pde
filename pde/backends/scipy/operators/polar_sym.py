r"""This module implements differential operators on polar grids.

.. autosummary::
   :nosignatures:

   make_poisson_solver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ....grids.spherical import PolarSymGrid
from ....tools.docstrings import fill_in_docstring
from .. import scipy_backend
from .common import make_general_poisson_solver

if TYPE_CHECKING:
    from ....grids.boundaries.axes import BoundariesList
    from ....tools.typing import NumericArray, OperatorImplType


@fill_in_docstring
def _get_laplace_matrix(bcs: BoundariesList) -> tuple[NumericArray, NumericArray]:
    """Get sparse matrix for laplace operator on a polar grid.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    assert isinstance(bcs.grid, PolarSymGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    scale = 1 / dr**2

    matrix = sparse.dok_matrix((dim_r, dim_r))
    vector = sparse.dok_matrix((dim_r, 1))

    for i in range(dim_r):
        matrix[i, i] += -2 * scale
        scale_i = 1 / (2 * rs[i] * dr)

        if i == 0:
            if r_min == 0:
                matrix[i, i + 1] = 2 * scale
                continue  # the special case of the inner boundary is handled
            const, entries = bcs[0].get_sparse_matrix_data((-1,))
            factor = scale - scale_i
            vector[i] += const * factor
            for k, v in entries.items():
                matrix[i, k] += v * factor

        else:
            matrix[i, i - 1] = scale - scale_i

        if i == dim_r - 1:
            const, entries = bcs[0].get_sparse_matrix_data((dim_r,))
            factor = scale + scale_i
            vector[i] += const * factor
            for k, v in entries.items():
                matrix[i, k] += v * factor

        else:
            matrix[i, i + 1] = scale + scale_i

    return matrix, vector


@scipy_backend.register_operator(PolarSymGrid, "poisson_solver", rank_in=0, rank_out=0)
@fill_in_docstring
def make_poisson_solver(
    bcs: BoundariesList, *, method: Literal["auto", "scipy"] = "auto"
) -> OperatorImplType:
    """Make a operator that solves Poisson's equation.

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str):
            The chosen method for implementing the operator

    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(bcs)
    return make_general_poisson_solver(matrix, vector, method)
