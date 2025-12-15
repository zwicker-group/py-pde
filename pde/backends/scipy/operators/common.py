"""Common functions that are used by many operators.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from ....grids.base import GridBase
    from ....tools.typing import NumericArray, OperatorImplType


def uniform_discretization(grid: GridBase) -> float:
    """Returns the uniform discretization or raises RuntimeError.

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
    msg = "Grid discretization is not uniform"
    raise RuntimeError(msg)


def make_laplace_from_matrix(
    matrix, vector
) -> Callable[[NumericArray, NumericArray | None], NumericArray]:
    """Make a Laplace operator using matrix vector products.

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

    def laplace(arr: NumericArray, out: NumericArray | None = None) -> NumericArray:
        """Apply the laplace operator to `arr`"""
        result = mat.dot(arr.flat) + vec
        if out is None:
            out = result.reshape(arr.shape)
        else:
            out[:] = result.reshape(arr.shape)
        return out

    return laplace


def make_general_poisson_solver(
    matrix, vector, method: Literal["auto", "scipy"] = "auto"
) -> OperatorImplType:
    """Make an operator that solves Poisson's problem.

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

    logger = logging.getLogger(__name__)

    if method not in {"auto", "scipy"}:
        msg = f"Method {method} is not available"
        raise ValueError(msg)

    # prepare the matrix representing the operator
    mat = matrix.tocsc()
    vec = vector.toarray()[:, 0]

    def solve_poisson(arr: NumericArray, out: NumericArray) -> None:
        """Solves Poisson's equation using sparse linear algebra."""
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
                msg = f"Poisson problem could not be solved (Residual: {residual})"
                raise RuntimeError(msg)
            logger.info("Solved Poisson problem with sparse.linalg.lsmr")

        # convert the result to the correct format
        out[:] = result.reshape(arr.shape)

    return solve_poisson
