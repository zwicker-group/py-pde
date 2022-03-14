r"""
This module implements differential operators on polar grids 

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_tensor_divergence
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Tuple

import numpy as np

from ...tools.docstrings import fill_in_docstring
from ...tools.numba import jit
from ...tools.typing import OperatorType
from ..boundaries import Boundaries
from ..spherical import PolarSymGrid
from .common import make_general_poisson_solver


@PolarSymGrid.register_operator("laplace", rank_in=0, rank_out=0)
@fill_in_docstring
def make_laplace(grid: PolarSymGrid) -> OperatorType:
    """make a discretized laplace operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]
    dr_2 = 1 / dr**2

    @jit
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """apply laplace operator to array `arr`"""
        for i in range(1, dim_r + 1):  # iterate inner radial points
            out[i - 1] = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * dr_2
            out[i - 1] += (arr[i + 1] - arr[i - 1]) / (2 * rs[i - 1] * dr)

    return laplace  # type: ignore


@PolarSymGrid.register_operator("gradient", rank_in=0, rank_out=1)
@fill_in_docstring
def make_gradient(grid: PolarSymGrid) -> OperatorType:
    """make a discretized gradient operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in range(1, dim_r + 1):  # iterate inner radial points
            out[0, i - 1] = (arr[i + 1] - arr[i - 1]) * scale_r
            out[1, i - 1] = 0  # no angular dependence by definition

    return gradient  # type: ignore


@PolarSymGrid.register_operator("gradient_squared", rank_in=0, rank_out=0)
@fill_in_docstring
def make_gradient_squared(grid: PolarSymGrid, central: bool = True) -> OperatorType:
    """make a discretized gradient squared operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]

    if central:
        # use central differences
        scale = 0.25 / dr**2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                out[i - 1] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

    else:
        # use forward and backward differences
        scale = 0.5 / dr**2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                term = (arr[i + 1] - arr[i]) ** 2 + (arr[i] - arr[i - 1]) ** 2
                out[i - 1] = term * scale

    return gradient_squared  # type: ignore


@PolarSymGrid.register_operator("divergence", rank_in=1, rank_out=0)
@fill_in_docstring
def make_divergence(grid: PolarSymGrid) -> OperatorType:
    """make a discretized divergence operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]
    scale_r = 1 / (2 * dr)

    @jit
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply divergence operator to array `arr`"""
        # inner radial boundary condition
        for i in range(1, dim_r + 1):  # iterate radial points
            out[i - 1] = (arr[0, i + 1] - arr[0, i - 1]) * scale_r
            out[i - 1] += arr[0, i] / rs[i - 1]

    return divergence  # type: ignore


@PolarSymGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
def make_vector_gradient(grid: PolarSymGrid) -> OperatorType:
    """make a discretized vector gradient operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply vector gradient operator to array `arr`"""
        # assign aliases
        arr_r, arr_φ = arr
        out_rr, out_rφ = out[0, 0, :], out[0, 1, :]
        out_φr, out_φφ = out[1, 0, :], out[1, 1, :]

        for i in range(1, dim_r + 1):  # iterate radial points
            out_rr[i - 1] = (arr_r[i + 1] - arr_r[i - 1]) * scale_r
            out_rφ[i - 1] = -arr_φ[i] / rs[i - 1]
            out_φr[i - 1] = (arr_φ[i + 1] - arr_φ[i - 1]) * scale_r
            out_φφ[i - 1] = arr_r[i] / rs[i - 1]

    return vector_gradient  # type: ignore


@PolarSymGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
@fill_in_docstring
def make_tensor_divergence(grid: PolarSymGrid) -> OperatorType:
    """make a discretized tensor divergence operator for a polar grid

    {DESCR_POLAR_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.PolarSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, PolarSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply tensor divergence operator to array `arr`"""
        # assign aliases
        arr_rr, arr_rφ = arr[0, 0, :], arr[0, 1, :]
        arr_φr, arr_φφ = arr[1, 0, :], arr[1, 1, :]
        out_r, out_φ = out[0, :], out[1, :]

        # iterate over inner points
        for i in range(1, dim_r + 1):
            term = (arr_rr[i] - arr_φφ[i]) / rs[i - 1]
            out_r[i - 1] = (arr_rr[i + 1] - arr_rr[i - 1]) * scale_r + term
            term = (arr_rφ[i] + arr_φr[i]) / rs[i - 1]
            out_φ[i - 1] = (arr_φr[i + 1] - arr_φr[i - 1]) * scale_r + term

    return tensor_divergence  # type: ignore


@fill_in_docstring
def _get_laplace_matrix(bcs: Boundaries) -> Tuple[np.ndarray, np.ndarray]:
    """get sparse matrix for laplace operator on a polar grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
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
            else:
                const, entries = bcs[0].get_data((-1,))
                factor = scale - scale_i
                vector[i] += const * factor
                for k, v in entries.items():
                    matrix[i, k] += v * factor

        else:
            matrix[i, i - 1] = scale - scale_i

        if i == dim_r - 1:
            const, entries = bcs[0].get_data((dim_r,))
            factor = scale + scale_i
            vector[i] += const * factor
            for k, v in entries.items():
                matrix[i, k] += v * factor

        else:
            matrix[i, i + 1] = scale + scale_i

    return matrix, vector


@PolarSymGrid.register_operator("poisson_solver", rank_in=0, rank_out=0)
@fill_in_docstring
def make_poisson_solver(bcs: Boundaries, method: str = "auto") -> OperatorType:
    """make a operator that solves Poisson's equation

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str):
            The chosen method for implementing the operator

    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(bcs)
    return make_general_poisson_solver(matrix, vector, method)
