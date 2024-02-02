"""
Common functions that are used by many operators 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Literal

import numpy as np

from ...tools.numba import jit
from ...tools.typing import OperatorType
from ..base import GridBase

logger = logging.getLogger(__name__)


def make_derivative(
    grid: GridBase,
    axis: int = 0,
    method: Literal["central", "forward", "backward"] = "central",
) -> OperatorType:
    """make a derivative operator along a single axis using numba compilation

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid for which the operator is created
        axis (int):
            The axis along which the derivative will be taken
        method (str):
            The method for calculating the derivative. Possible values are
            'central', 'forward', and 'backward'.

    Returns:
        A function that can be applied to an full array of values including those at
        ghost cells. The result will be an array of the same shape containing the actual
        derivatives at the valid (interior) grid points.
    """
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Unknown derivative type `{method}`")

    shape = grid.shape
    num_axes = len(shape)
    dx = grid.discretization[axis]

    if axis == 0:
        di, dj, dk = 1, 0, 0
    elif axis == 1:
        di, dj, dk = 0, 1, 0
    elif axis == 2:
        di, dj, dk = 0, 0, 1
    else:
        raise NotImplementedError(f"Derivative for {axis:d} dimensions")

    if num_axes == 1:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 1d array `arr`"""
            for i in range(1, shape[0] + 1):
                if method == "central":
                    out[i - 1] = (arr[i + 1] - arr[i - 1]) / (2 * dx)
                elif method == "forward":
                    out[i - 1] = (arr[i + 1] - arr[i]) / dx
                elif method == "backward":
                    out[i - 1] = (arr[i] - arr[i - 1]) / dx

    elif num_axes == 2:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 2d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    arr_l = arr[i - di, j - dj]
                    arr_r = arr[i + di, j + dj]
                    if method == "central":
                        out[i - 1, j - 1] = (arr_r - arr_l) / (2 * dx)
                    elif method == "forward":
                        out[i - 1, j - 1] = (arr_r - arr[i, j]) / dx
                    elif method == "backward":
                        out[i - 1, j - 1] = (arr[i, j] - arr_l) / dx

    elif num_axes == 3:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 3d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    for k in range(1, shape[2] + 1):
                        arr_l = arr[i - di, j - dj, k - dk]
                        arr_r = arr[i + di, j + dj, k + dk]
                        if method == "central":
                            out[i - 1, j - 1, k - 1] = (arr_r - arr_l) / (2 * dx)
                        elif method == "forward":
                            out[i - 1, j - 1, k - 1] = (arr_r - arr[i, j, k]) / dx
                        elif method == "backward":
                            out[i - 1, j - 1, k - 1] = (arr[i, j, k] - arr_l) / dx

    else:
        raise NotImplementedError(
            f"Numba derivative operator not implemented for {num_axes:d} axes"
        )

    return diff  # type: ignore


def make_derivative2(grid: GridBase, axis: int = 0) -> OperatorType:
    """make a second-order derivative operator along a single axis

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid for which the operator is created
        axis (int):
            The axis along which the derivative will be taken

    Returns:
        A function that can be applied to an full array of values including those at
        ghost cells. The result will be an array of the same shape containing the actual
        derivatives at the valid (interior) grid points.
    """
    shape = grid.shape
    num_axes = len(shape)
    scale = 1 / grid.discretization[axis] ** 2

    if axis == 0:
        di, dj, dk = 1, 0, 0
    elif axis == 1:
        di, dj, dk = 0, 1, 0
    elif axis == 2:
        di, dj, dk = 0, 0, 1
    else:
        raise NotImplementedError(f"Derivative for {axis:d} dimensions")

    if num_axes == 1:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 1d array `arr`"""
            for i in range(1, shape[0] + 1):
                out[i - 1] = (arr[i + 1] - 2 * arr[i] + arr[i - 1]) * scale

    elif num_axes == 2:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 2d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    arr_l = arr[i - di, j - dj]
                    arr_r = arr[i + di, j + dj]
                    out[i - 1, j - 1] = (arr_r - 2 * arr[i, j] + arr_l) * scale

    elif num_axes == 3:

        @jit
        def diff(arr: np.ndarray, out: np.ndarray) -> None:
            """calculate derivative of 3d array `arr`"""
            for i in range(1, shape[0] + 1):
                for j in range(1, shape[1] + 1):
                    for k in range(1, shape[2] + 1):
                        arr_l = arr[i - di, j - dj, k - dk]
                        arr_r = arr[i + di, j + dj, k + dk]
                        out[i - 1, j - 1, k - 1] = (
                            arr_r - 2 * arr[i, j, k] + arr_l
                        ) * scale

    else:
        raise NotImplementedError(
            f"Numba derivative operator not implemented for {num_axes:d} axes"
        )

    return diff  # type: ignore


def uniform_discretization(grid: GridBase) -> float:
    """returns the uniform discretization or raises RuntimeError

    Args:
        grid (:class:`~pde.grids.base.GridBase`):
            The grid whose discretization is tested

    Raises:
        `RuntimeError` if discretization is different between different axes

    Returns:
        float: the common discretization of all axes
    """
    dx_mean = np.mean(grid.discretization)
    if np.allclose(grid.discretization, dx_mean):
        return float(dx_mean)
    else:
        raise RuntimeError("Grid discretization is not uniform")


def make_laplace_from_matrix(
    matrix, vector
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """make a Laplace operator using matrix vector products

    Args:
        matrix:
            (Sparse) matrix representing the laplace operator on the given grid
        vector:
            Constant part representing the boundary conditions of the Laplace operator

    Returns:
        A function that can be applied to an array of values to obtain the result of
        applying the linear operator `matrix` and the offset given by `vector`.
    """
    mat = matrix.tocsc()
    vec = vector.toarray()[:, 0]

    def laplace(arr: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """apply the laplace operator to `arr`"""
        result = mat.dot(arr.flat) + vec
        if out is None:
            out = result.reshape(arr.shape)
        else:
            out[:] = result.reshape(arr.shape)
        return out

    return laplace


def make_general_poisson_solver(
    matrix, vector, method: Literal["auto", "scipy"] = "auto"
) -> OperatorType:
    """make an operator that solves Poisson's problem

    Args:
        matrix:
            The (sparse) matrix representing the laplace operator on the given grid.
        vector:
            The constant part representing the boundary conditions of the Laplace
            operator.
        method (str):
            The chosen method for implementing the operator

    Returns:
        A function that can be applied to an array of values to obtain the solution to
        Poisson's equation where the array is used as the right hand side
    """
    from scipy import sparse

    try:
        from scipy.sparse.linalg import MatrixRankWarning
    except ImportError:
        from scipy.sparse.linalg.dsolve.linsolve import MatrixRankWarning

    if method not in {"auto", "scipy"}:
        raise ValueError(f"Method {method} is not available")

    # prepare the matrix representing the operator
    mat = matrix.tocsc()
    vec = vector.toarray()[:, 0]

    def solve_poisson(arr: np.ndarray, out: np.ndarray) -> None:
        """solves Poisson's equation using sparse linear algebra"""
        # prepare the right hand side vector
        rhs = np.ravel(arr) - vec

        # solve the linear problem using a sparse solver
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # enable warning catching
                result = sparse.linalg.spsolve(mat, rhs)

        except MatrixRankWarning:
            # this can happen for singular laplace matrix, e.g. when pure
            # Neumann conditions are considered. In this case, a solution is
            # obtained using least squares
            logger.warning(
                "Poisson problem seems to be under-determined and "
                "could not be solved using sparse.linalg.spsolve"
            )
            use_leastsquares = True

        else:
            # test whether the solution is good enough
            if np.allclose(mat.dot(result), rhs, rtol=1e-5, atol=1e-5):
                logger.info("Solved Poisson problem with sparse.linalg.spsolve")
                use_leastsquares = False
            else:
                logger.warning(
                    "Poisson problem was not solved using sparse.linalg.spsolve"
                )
                use_leastsquares = True

        if use_leastsquares:
            # use least squares to solve an underdetermined problem
            result = sparse.linalg.lsmr(mat, rhs)[0]
            if not np.allclose(mat.dot(result), rhs, rtol=1e-5, atol=1e-5):
                residual = np.linalg.norm(mat.dot(result) - rhs)
                raise RuntimeError(
                    f"Poisson problem could not be solved (Residual: {residual})"
                )
            logger.info("Solved Poisson problem with sparse.linalg.lsmr")

        # convert the result to the correct format
        out[:] = result.reshape(arr.shape)

    return solve_poisson
