r"""
This module implements differential operators on spherical grids 

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
from ..spherical import SphericalSymGrid
from .common import make_general_poisson_solver


@SphericalSymGrid.register_operator("laplace", rank_in=0, rank_out=0)
@fill_in_docstring
def make_laplace(grid: SphericalSymGrid, conservative: bool = True) -> OperatorType:
    """make a discretized laplace operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        conservative (bool):
            Flag indicating whether the laplace operator should be conservative (which
            results in slightly slower computations).

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]
    r_min, r_max = grid.axes_bounds[0]

    if conservative:
        # create a conservative spherical laplace operator
        rl = rs - dr / 2  # inner radii of spherical shells
        rh = rs + dr / 2  # outer radii
        assert np.isclose(rl[0], r_min) and np.isclose(rh[-1], r_max)
        volumes = (rh ** 3 - rl ** 3) / 3  # volume of the spherical shells
        factor_l = (rs - 0.5 * dr) ** 2 / (dr * volumes)
        factor_h = (rs + 0.5 * dr) ** 2 / (dr * volumes)

        @jit
        def laplace(arr: np.ndarray, out: np.ndarray) -> None:
            """apply laplace operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                out[i - 1] = factor_h[i - 1] * (arr[i + 1] - arr[i])
                out[i - 1] -= factor_l[i - 1] * (arr[i] - arr[i - 1])

    else:  # create an operator that is not conservative
        dr2 = 1 / dr ** 2

        @jit
        def laplace(arr: np.ndarray, out: np.ndarray) -> None:
            """apply laplace operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                out[i - 1] = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * dr2
                out[i - 1] += (arr[i + 1] - arr[i - 1]) / (rs[i - 1] * dr)

    return laplace  # type: ignore


@SphericalSymGrid.register_operator("gradient", rank_in=0, rank_out=1)
@fill_in_docstring
def make_gradient(grid: SphericalSymGrid) -> OperatorType:
    """make a discretized gradient operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]

    scale_r = 1 / (2 * dr)

    @jit
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in range(1, dim_r + 1):  # iterate inner radial points
            out[0, i - 1] = (arr[i + 1] - arr[i - 1]) * scale_r
            out[1, i - 1] = out[2, i - 1] = 0  # no angular dependence by definition

    return gradient  # type: ignore


@SphericalSymGrid.register_operator("gradient_squared", rank_in=0, rank_out=0)
@fill_in_docstring
def make_gradient_squared(grid: SphericalSymGrid, central: bool = True) -> OperatorType:
    """make a discretized gradient squared operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]

    if central:
        # use central differences
        scale = 0.25 / dr ** 2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                out[i - 1] = (arr[i + 1] - arr[i - 1]) ** 2 * scale

    else:
        # use forward and backward differences
        scale = 0.5 / dr ** 2

        @jit
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply squared gradient operator to array `arr`"""
            for i in range(1, dim_r + 1):  # iterate inner radial points
                term = (arr[i + 1] - arr[i]) ** 2 + (arr[i] - arr[i - 1]) ** 2
                out[i - 1] = term * scale

    return gradient_squared  # type: ignore


@SphericalSymGrid.register_operator("divergence", rank_in=1, rank_out=0)
@fill_in_docstring
def make_divergence(grid: SphericalSymGrid, safe: bool = True) -> OperatorType:
    """make a discretized divergence operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Warning:
        This operator ignores the θ-component of the field when calculating the
        divergence. This is because the resulting scalar field could not be expressed
        on a :class:`~pde.grids.spherical_sym.SphericalSymGrid`.

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        safe (bool):
            Add extra checks for the validity of the input

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    dr = grid.discretization[0]
    rs = grid.axes_coords[0]

    scale_r = 1 / (2 * dr)
    fs = 2 / rs  # factors that need to be multiplied below

    @jit
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply divergence operator to array `arr`"""
        if safe:
            assert np.all(arr[1, 1:-1] == 0)
        arr_r = arr[0, :]
        for i in range(1, dim_r + 1):  # iterate radial points
            out[i - 1] = (arr_r[i + 1] - arr_r[i - 1]) * scale_r + fs[i - 1] * arr_r[i]

    return divergence  # type: ignore


@SphericalSymGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
def make_vector_gradient(grid: SphericalSymGrid, safe: bool = True) -> OperatorType:
    """make a discretized vector gradient operator for a spherical grid

    Warning:
        This operator ignores the two angular components of the field when calculating
        the gradient. This is because the resulting field could not be expressed on a
        :class:`~pde.grids.spherical_sym.SphericalSymGrid`.

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        safe (bool):
            Add extra checks for the validity of the input

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply vector gradient operator to array `arr`"""
        if safe:
            assert np.all(arr[1:, 1:-1] == 0)

        # assign aliases
        arr_r = arr[0, :]
        out_rr, out_rθ, out_rφ = out[0, 0, :], out[0, 1, :], out[0, 2, :]
        out_θr, out_θθ, out_θφ = out[1, 0, :], out[1, 1, :], out[1, 2, :]
        out_φr, out_φθ, out_φφ = out[2, 0, :], out[2, 1, :], out[2, 2, :]

        # set all components to zero that are not affected
        out_rθ[:] = 0
        out_rφ[:] = 0
        out_θr[:] = 0
        out_θφ[:] = 0
        out_φr[:] = 0
        out_φθ[:] = 0

        # inner radial boundary condition
        for i in range(1, dim_r + 1):  # iterate radial points
            out_rr[i - 1] = (arr_r[i + 1] - arr_r[i - 1]) * scale_r
            out_θθ[i - 1] = arr_r[i] / rs[i - 1]
            out_φφ[i - 1] = arr_r[i] / rs[i - 1]

    return vector_gradient  # type: ignore


@SphericalSymGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
@fill_in_docstring
def make_tensor_divergence(grid: SphericalSymGrid, safe: bool = True) -> OperatorType:
    """make a discretized tensor divergence operator for a spherical grid

    {DESCR_SPHERICAL_GRID}

    Args:
        grid (:class:`~pde.grids.spherical.SphericalSymGrid`):
            The polar grid for which this operator will be defined
        safe (bool):
            Add extra checks for the validity of the input

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, SphericalSymGrid)

    # calculate preliminary quantities
    dim_r = grid.shape[0]
    rs = grid.axes_coords[0]
    dr = grid.discretization[0]
    scale_r = 1 / (2 * dr)

    @jit
    def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply tensor divergence operator to array `arr`"""
        # assign aliases
        arr_rr, arr_rθ, arr_rφ = arr[0, 0, :], arr[0, 1, :], arr[0, 2, :]
        arr_θr, arr_θθ, arr_θφ = arr[1, 0, :], arr[1, 1, :], arr[1, 2, :]
        arr_φr, arr_φθ, arr_φφ = arr[2, 0, :], arr[2, 1, :], arr[2, 2, :]
        out_r, out_θ, out_φ = out[0, :], out[1, :], out[2, :]

        # check inputs
        if safe:
            assert np.all(arr_rθ[1:-1] == 0)
            assert np.all(arr_θθ[1:-1] == 0)
            assert np.all(arr_φφ[1:-1] == 0)
            assert np.all(arr_φθ[1:-1] == 0)
            assert np.all(arr_θφ[1:-1] == 0)

        # iterate over inner points
        for i in range(1, dim_r + 1):
            deriv_r = (arr_rr[i + 1] - arr_rr[i - 1]) * scale_r
            out_r[i - 1] = deriv_r + 2 * arr_rr[i] / rs[i - 1]

            deriv_r = (arr_θr[i + 1] - arr_θr[i - 1]) * scale_r
            out_θ[i - 1] = deriv_r + 2 * arr_θr[i] / rs[i - 1]

            deriv_r = (arr_φr[i + 1] - arr_φr[i - 1]) * scale_r
            out_φ[i - 1] = deriv_r + (2 * arr_φr[i] + arr_rφ[i]) / rs[i - 1]

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
    volumes = (rh ** 3 - rl ** 3) / 3  # volume of the spherical shells

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
                const, entries = bcs[0].get_data((-1,))
                vector[i] += const * factor_l[i]
                for k, v in entries.items():
                    matrix[i, k] += v * factor_l[i]

        else:
            matrix[i, i - 1] = factor_l[i]

        if i == dim_r - 1:
            const, entries = bcs[0].get_data((dim_r,))
            vector[i] += const * factor_h[i]
            for k, v in entries.items():
                matrix[i, k] += v * factor_h[i]

        else:
            matrix[i, i + 1] = factor_h[i]

    return matrix, vector


@SphericalSymGrid.register_operator("poisson_solver", rank_in=0, rank_out=0)
@fill_in_docstring
def make_poisson_solver(bcs: Boundaries, method: str = "auto") -> OperatorType:
    """make a operator that solves Poisson's equation

    {DESCR_POLAR_GRID}

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}

    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(bcs)
    return make_general_poisson_solver(matrix, vector, method)
