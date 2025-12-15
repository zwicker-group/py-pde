r"""This module implements differential operators on cylindrical grids.

.. autosummary::
   :nosignatures:

   make_poisson_solver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ....grids.cylindrical import CylindricalSymGrid
from ....tools.docstrings import fill_in_docstring
from .. import scipy_backend
from .common import make_general_poisson_solver

if TYPE_CHECKING:
    from ....grids.boundaries.axes import BoundariesList
    from ....tools.typing import NumericArray, OperatorImplType


def _get_laplace_matrix(bcs: BoundariesList) -> tuple[NumericArray, NumericArray]:
    """Get sparse matrix for Laplace operator on a cylindrical grid.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate the
        discretized laplacian
    """
    from scipy import sparse

    grid = bcs.grid
    assert isinstance(grid, CylindricalSymGrid)
    dim_r, dim_z = grid.shape
    matrix = sparse.dok_matrix((dim_r * dim_z, dim_r * dim_z))
    vector = sparse.dok_matrix((dim_r * dim_z, 1))

    bc_r, bc_z = bcs
    scale_r, scale_z = grid.discretization**-2
    factor_r = 1 / (2 * grid.axes_coords[0] * grid.discretization[0])

    def i(r, z):
        """Helper function for flattening the index.

        This is equivalent to np.ravel_multi_index((r, z), (dim_r, dim_z))
        """
        return r * dim_z + z

    # set diagonal elements, i.e., the central value in the kernel
    matrix.setdiag(-2 * (scale_r + scale_z))

    for r in range(dim_r):
        for z in range(dim_z):
            # handle r-direction
            if r == 0:
                const, entries = bc_r.get_sparse_matrix_data((-1, z))
                vector[i(r, z)] += const * (scale_r - factor_r[0])
                for k, v in entries.items():
                    matrix[i(r, z), i(k, z)] += v * (scale_r - factor_r[0])
            else:
                matrix[i(r, z), i(r - 1, z)] += scale_r - factor_r[r]

            if r == dim_r - 1:
                const, entries = bc_r.get_sparse_matrix_data((dim_r, z))
                vector[i(r, z)] += const * (scale_r + factor_r[-1])
                for k, v in entries.items():
                    matrix[i(r, z), i(k, z)] += v * (scale_r + factor_r[-1])
            else:
                matrix[i(r, z), i(r + 1, z)] += scale_r + factor_r[r]

            # handle z-direction
            if z == 0:
                const, entries = bc_z.get_sparse_matrix_data((r, -1))
                vector[i(r, z)] += const * scale_z
                for k, v in entries.items():
                    matrix[i(r, z), i(r, k)] += v * scale_z
            else:
                matrix[i(r, z), i(r, z - 1)] += scale_z

            if z == dim_z - 1:
                const, entries = bc_z.get_sparse_matrix_data((r, dim_z))
                vector[i(r, z)] += const * scale_z
                for k, v in entries.items():
                    matrix[i(r, z), i(r, k)] += v * scale_z
            else:
                matrix[i(r, z), i(r, z + 1)] += scale_z

    return matrix, vector


@scipy_backend.register_operator(
    CylindricalSymGrid, "poisson_solver", rank_in=0, rank_out=0
)
@fill_in_docstring
def make_poisson_solver(
    bcs: BoundariesList, *, method: Literal["auto", "scipy"] = "auto"
) -> OperatorImplType:
    """Make a operator that solves Poisson's equation.

    {DESCR_CYLINDRICAL_GRID}

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
