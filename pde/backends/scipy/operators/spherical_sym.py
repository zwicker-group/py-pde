r"""This module implements differential operators on spherical grids.

.. autosummary::
   :nosignatures:

   make_poisson_solver

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ....grids.spherical import SphericalSymGrid
from ....tools.docstrings import fill_in_docstring
from .. import scipy_backend
from .common import make_general_poisson_solver

if TYPE_CHECKING:
    from ....grids.boundaries.axes import BoundariesList
    from ....tools.typing import NumericArray, OperatorImplType


@fill_in_docstring
def _get_laplace_matrix(bcs: BoundariesList) -> tuple[NumericArray, NumericArray]:
    """Get sparse matrix for laplace operator on a spherical grid.

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.BoundariesList`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """
    from scipy import sparse

    assert isinstance(bcs.grid, SphericalSymGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, r_max = bcs.grid.axes_bounds[0]

    # create a conservative spherical laplace operator
    rl = r_min + dr * np.arange(dim_r)  # inner radii of spherical shells
    rh = rl + dr  # outer radii
    assert np.isclose(rh[-1], r_max)
    volumes = (rh**3 - rl**3) / 3  # volume of the spherical shells

    factor_l = (rs - 0.5 * dr) ** 2 / (dr * volumes)
    factor_h = (rs + 0.5 * dr) ** 2 / (dr * volumes)

    matrix = sparse.dok_matrix((dim_r, dim_r))
    vector = sparse.dok_matrix((dim_r, 1))

    for i in range(dim_r):
        matrix[i, i] += -factor_l[i] - factor_h[i]

        if i == 0:
            if r_min == 0:
                matrix[i, i + 1] = factor_l[i]
            else:
                const, entries = bcs[0].get_sparse_matrix_data((-1,))
                vector[i] += const * factor_l[i]
                for k, v in entries.items():
                    matrix[i, k] += v * factor_l[i]

        else:
            matrix[i, i - 1] = factor_l[i]

        if i == dim_r - 1:
            const, entries = bcs[0].get_sparse_matrix_data((dim_r,))
            vector[i] += const * factor_h[i]
            for k, v in entries.items():
                matrix[i, k] += v * factor_h[i]

        else:
            matrix[i, i + 1] = factor_h[i]

    return matrix, vector


@scipy_backend.register_operator(
    SphericalSymGrid, "poisson_solver", rank_in=0, rank_out=0
)
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
