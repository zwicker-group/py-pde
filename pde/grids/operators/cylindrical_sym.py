r"""
This module implements differential operators on cylindrical grids 

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_vector_laplace
   make_tensor_divergence
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np

from ... import config
from ...tools.docstrings import fill_in_docstring
from ...tools.numba import jit
from ...tools.typing import OperatorType
from ..cylindrical import CylindricalSymGrid


@CylindricalSymGrid.register_operator("laplace", rank_in=0, rank_out=0)
@fill_in_docstring
def make_laplace(grid: CylindricalSymGrid) -> OperatorType:
    """make a discretized laplace operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    dim_r, dim_z = grid.shape
    dr_2, dz_2 = 1 / grid.discretization**2

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """apply laplace operator to array `arr`"""
        for i in nb.prange(1, dim_r + 1):  # iterate radial points
            for j in range(1, dim_z + 1):  # iterate axial points
                arr_z_l, arr_z_h = arr[i, j - 1], arr[i, j + 1]
                arr_r_l, arr_r_h = arr[i - 1, j], arr[i + 1, j]
                out[i - 1, j - 1] = (
                    (arr_r_h - 2 * arr[i, j] + arr_r_l) * dr_2
                    + (arr_r_h - arr_r_l) / (2 * i - 1) * dr_2
                    + (arr_z_l - 2 * arr[i, j] + arr_z_h) * dz_2
                )

    return laplace  # type: ignore


@CylindricalSymGrid.register_operator("gradient", rank_in=0, rank_out=1)
@fill_in_docstring
def make_gradient(grid: CylindricalSymGrid) -> OperatorType:
    """make a discretized gradient operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    dim_r, dim_z = grid.shape
    scale_r, scale_z = 1 / (2 * grid.discretization)

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        for i in nb.prange(1, dim_r + 1):  # iterate radial points
            for j in range(1, dim_z + 1):  # iterate axial points
                out[0, i - 1, j - 1] = (arr[i + 1, j] - arr[i - 1, j]) * scale_r
                out[1, i - 1, j - 1] = (arr[i, j + 1] - arr[i, j - 1]) * scale_z
                out[2, i - 1, j - 1] = 0  # no phi dependence by definition

    return gradient  # type: ignore


@CylindricalSymGrid.register_operator("gradient_squared", rank_in=0, rank_out=0)
@fill_in_docstring
def make_gradient_squared(
    grid: CylindricalSymGrid, central: bool = True
) -> OperatorType:
    """make a discretized gradient squared operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created
        central (bool):
            Whether a central difference approximation is used for the gradient
            operator. If this is False, the squared gradient is calculated as
            the mean of the squared values of the forward and backward
            derivatives.

    Returns:
        A function that can be applied to an array of values
    """
    # use processing for large enough arrays
    dim_r, dim_z = grid.shape
    parallel = dim_r * dim_z >= config["numba.multithreading_threshold"]

    if central:
        # use central differences
        scale_r, scale_z = 0.25 / grid.discretization**2

        @jit(parallel=parallel)
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply gradient operator to array `arr`"""
            for i in nb.prange(1, dim_r + 1):  # iterate radial points
                for j in range(1, dim_z + 1):  # iterate axial points
                    term_r = (arr[i + 1, j] - arr[i - 1, j]) ** 2
                    term_z = (arr[i, j + 1] - arr[i, j - 1]) ** 2
                    out[i - 1, j - 1] = term_r * scale_r + term_z * scale_z

    else:
        # use forward and backward differences
        scale_r, scale_z = 0.5 / grid.discretization**2

        @jit(parallel=parallel)
        def gradient_squared(arr: np.ndarray, out: np.ndarray) -> None:
            """apply gradient operator to array `arr`"""
            for i in nb.prange(1, dim_r + 1):  # iterate radial points
                for j in range(1, dim_z + 1):  # iterate axial points
                    arr_z_l, arr_c, arr_z_h = arr[i, j - 1], arr[i, j], arr[i, j + 1]
                    term_r = (arr[i + 1, j] - arr_c) ** 2 + (arr_c - arr[i - 1, j]) ** 2
                    term_z = (arr_z_h - arr_c) ** 2 + (arr_c - arr_z_l) ** 2
                    out[i - 1, j - 1] = term_r * scale_r + term_z * scale_z

    return gradient_squared  # type: ignore


@CylindricalSymGrid.register_operator("divergence", rank_in=1, rank_out=0)
@fill_in_docstring
def make_divergence(grid: CylindricalSymGrid) -> OperatorType:
    """make a discretized divergence operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    dim_r, dim_z = grid.shape
    scale_r, scale_z = 1 / (2 * grid.discretization)
    rs = grid.axes_coords[0]

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply divergence operator to array `arr`"""
        arr_r, arr_z = arr[0], arr[1]

        for i in nb.prange(1, dim_r + 1):  # iterate radial points
            for j in range(1, dim_z + 1):  # iterate axial points
                out[i - 1, j - 1] = (
                    arr_r[i, j] / rs[i - 1]
                    + (arr_r[i + 1, j] - arr_r[i - 1, j]) * scale_r
                    + (arr_z[i, j + 1] - arr_z[i, j - 1]) * scale_z
                )

    return divergence  # type: ignore


@CylindricalSymGrid.register_operator("vector_gradient", rank_in=1, rank_out=2)
@fill_in_docstring
def make_vector_gradient(grid: CylindricalSymGrid) -> OperatorType:
    """make a discretized vector gradient operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    dim_r, dim_z = grid.shape
    scale_r, scale_z = 1 / (2 * grid.discretization)
    rs = grid.axes_coords[0]

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def vector_gradient(arr: np.ndarray, out: np.ndarray) -> None:
        """apply gradient operator to array `arr`"""
        # assign aliases
        arr_r, arr_z, arr_φ = arr
        out_rr, out_rz, out_rφ = out[0, 0], out[0, 1], out[0, 2]
        out_zr, out_zz, out_zφ = out[1, 0], out[1, 1], out[1, 2]
        out_φr, out_φz, out_φφ = out[2, 0], out[2, 1], out[2, 2]

        for i in nb.prange(1, dim_r + 1):  # iterate radial points
            for j in range(1, dim_z + 1):  # iterate axial points
                out_rr[i - 1, j - 1] = (arr_r[i + 1, j] - arr_r[i - 1, j]) * scale_r
                out_φr[i - 1, j - 1] = (arr_φ[i + 1, j] - arr_φ[i - 1, j]) * scale_r
                out_zr[i - 1, j - 1] = (arr_z[i + 1, j] - arr_z[i - 1, j]) * scale_r

                out_rφ[i - 1, j - 1] = -arr_φ[i, j] / rs[i - 1]
                out_φφ[i - 1, j - 1] = arr_r[i, j] / rs[i - 1]
                out_zφ[i - 1, j - 1] = 0

                out_rz[i - 1, j - 1] = (arr_r[i, j + 1] - arr_r[i, j - 1]) * scale_z
                out_φz[i - 1, j - 1] = (arr_φ[i, j + 1] - arr_φ[i, j - 1]) * scale_z
                out_zz[i - 1, j - 1] = (arr_z[i, j + 1] - arr_z[i, j - 1]) * scale_z

    return vector_gradient  # type: ignore


@CylindricalSymGrid.register_operator("vector_laplace", rank_in=1, rank_out=1)
@fill_in_docstring
def make_vector_laplace(grid: CylindricalSymGrid) -> OperatorType:
    """make a discretized vector laplace operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """

    # calculate preliminary quantities
    dim_r, dim_z = grid.shape
    rs = grid.axes_coords[0]
    dr, dz = grid.discretization
    s1, s2 = 1 / (2 * dr), 1 / dr**2
    scale_z = 1 / (dz**2)

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def vector_laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """apply vector laplace operator to array `arr`"""
        # assign aliases
        arr_r, arr_z, arr_φ = arr
        out_r, out_z, out_φ = out

        for i in nb.prange(1, dim_r + 1):  # iterate radial points
            for j in range(1, dim_z + 1):  # iterate axial points
                f_r_l, f_r_m, f_r_h = arr_r[i - 1, j], arr_r[i, j], arr_r[i + 1, j]
                out_r[i - 1, j - 1] = (
                    (arr_r[i, j + 1] - 2 * f_r_m + arr_r[i, j - 1]) * scale_z
                    - f_r_m / rs[i - 1] ** 2
                    + (f_r_h - f_r_l) * s1 / rs[i - 1]
                    + (f_r_h - 2 * f_r_m + f_r_l) * s2
                )

                f_φ_l, f_φ_m, f_φ_h = arr_φ[i - 1, j], arr_φ[i, j], arr_φ[i + 1, j]
                out_φ[i - 1, j - 1] = (
                    (arr_φ[i, j + 1] - 2 * f_φ_m + arr_φ[i, j - 1]) * scale_z
                    - f_φ_m / rs[i - 1] ** 2
                    + (f_φ_h - f_φ_l) * s1 / rs[i - 1]
                    + (f_φ_h - 2 * f_φ_m + f_φ_l) * s2
                )

                f_z_l, f_z_m, f_z_h = arr_z[i - 1, j], arr_z[i, j], arr_z[i + 1, j]
                out_z[i - 1, j - 1] = (
                    (arr_z[i, j + 1] - 2 * f_z_m + arr_z[i, j - 1]) * scale_z
                    + (f_z_h - f_z_l) * s1 / rs[i - 1]
                    + (f_z_h - 2 * f_z_m + f_z_l) * s2
                )

    return vector_laplace  # type: ignore


@CylindricalSymGrid.register_operator("tensor_divergence", rank_in=2, rank_out=1)
@fill_in_docstring
def make_tensor_divergence(grid: CylindricalSymGrid) -> OperatorType:
    """make a discretized tensor divergence operator for a cylindrical grid

    {DESCR_CYLINDRICAL_GRID}

    Args:
        grid (:class:`~pde.grids.cylindrical.CylindricalSymGrid`):
            The grid for which the operator is created

    Returns:
        A function that can be applied to an array of values
    """
    # calculate preliminary quantities
    dim_r, dim_z = grid.shape
    rs = grid.axes_coords[0]
    scale_r, scale_z = 1 / (2 * grid.discretization)

    # use processing for large enough arrays
    parallel = dim_r * dim_z >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def tensor_divergence(arr: np.ndarray, out: np.ndarray) -> None:
        """apply tensor divergence operator to array `arr`"""
        # assign aliases
        arr_rr, arr_rz, arr_rφ = arr[0, 0], arr[0, 1], arr[0, 2]
        arr_zr, arr_zz, _ = arr[1, 0], arr[1, 1], arr[1, 2]
        arr_φr, arr_φz, arr_φφ = arr[2, 0], arr[2, 1], arr[2, 2]
        out_r, out_z, out_φ = out

        for i in nb.prange(1, dim_r + 1):  # iterate radial points
            for j in range(1, dim_z + 1):  # iterate axial points
                out_r[i - 1, j - 1] = (
                    (arr_rz[i, j + 1] - arr_rz[i, j - 1]) * scale_z
                    + (arr_rr[i + 1, j] - arr_rr[i - 1, j]) * scale_r
                    + (arr_rr[i, j] - arr_φφ[i, j]) / rs[i - 1]
                )

                out_φ[i - 1, j - 1] = (
                    (arr_φz[i, j + 1] - arr_φz[i, j - 1]) * scale_z
                    + (arr_φr[i + 1, j] - arr_φr[i - 1, j]) * scale_r
                    + (arr_rφ[i, j] + arr_φr[i, j]) / rs[i - 1]
                )

                out_z[i - 1, j - 1] = (
                    (arr_zz[i, j + 1] - arr_zz[i, j - 1]) * scale_z
                    + (arr_zr[i + 1, j] - arr_zr[i - 1, j]) * scale_r
                    + arr_zr[i, j] / rs[i - 1]
                )

    return tensor_divergence  # type: ignore
