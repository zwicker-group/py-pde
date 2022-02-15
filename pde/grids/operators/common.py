"""
Common functions that are used by many operators 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import warnings

import numpy as np

from ...tools.typing import OperatorType
from ..base import GridBase

logger = logging.getLogger(__name__)


def uniform_discretization(grid: GridBase) -> float:
    """returns the uniform discretization or raises RuntimeError"""
    dx_mean = np.mean(grid.discretization)
    if np.allclose(grid.discretization, dx_mean):
        return float(dx_mean)
    else:
        raise RuntimeError("Grid discretization is not uniform")


def make_laplace_from_matrix(matrix, vector) -> OperatorType:
    """make a Laplace operator using matrix vector products

    Args:
        matrix:
            The (sparse) matrix representing the laplace operator on the given
            grid.
        vector:
            The constant part representing the boundary conditions of the
            Laplace operator.

    Returns:
        A function that can be applied to an array of values to obtain the
        solution to Poisson's equation where the array is used as the right hand
        side
    """
    mat = matrix.tocsc()
    vec = vector.toarray()[:, 0]

    def laplace(arr: np.ndarray, out: np.ndarray) -> None:
        """apply the laplace operator to `arr`"""
        result = mat.dot(arr.flat) + vec
        out[:] = result.reshape(arr.shape)

    return laplace


def make_general_poisson_solver(matrix, vector, method: str = "auto") -> OperatorType:
    """make an operator that solves Poisson's problem

    Args:
        matrix:
            The (sparse) matrix representing the laplace operator on the given
            grid.
        vector:
            The constant part representing the boundary conditions of the
            Laplace operator.
        method (str):
            The chosen method for implementing the operator

    Returns:
        A function that can be applied to an array of values to obtain the
        solution to Poisson's equation where the array is used as the right hand
        side
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
