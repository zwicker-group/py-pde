'''
Common functions that are used by many operators 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from typing import Callable
import warnings

import numpy as np
from scipy import sparse



def make_laplace_from_matrix(matrix, vector) -> Callable:
    """ make a Laplace operator using matrix vector products

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
    
    def laplace(arr: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """ apply the laplace operator to `arr` """
        result = mat.dot(arr.flat) + vec
        if out is None:
            return result.reshape(arr.shape)
        else:
            out.flat[:] = result
            return out    
    
    return laplace



def make_poisson_solver(matrix, vector, method: str = 'auto') -> Callable:
    """ make an operator that solves Poisson's problem

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
    if method not in {'auto', 'scipy'}:
        raise ValueError(f'Method {method} is not available')
    
    # prepare the matrix representing the operator
    mat = matrix.tocsc()
    vec = vector.toarray()[:, 0]
    
    def solve_poisson(arr: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        """ solves Poisson's equation using sparse linear algebra """
        # prepare the right hand side vector
        rhs = arr.flat - vec
        
        # solve the linear problem using a sparse solver
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # enable warning catching
            try:
                result = sparse.linalg.spsolve(mat, rhs)
            except sparse.linalg.dsolve.linsolve.MatrixRankWarning:
                # this can happen for singular laplace matrix, e.g. when pure
                # Neumann conditions are considered. In this case, a solution is
                # obtained using least squares 
                result = sparse.linalg.lsmr(mat, rhs)[0]
        
        # convert the result to the correct format
        if out is not None:
            out.flat[:] = result
        else:
            out = result.reshape(arr.shape)
        return out
    
    return solve_poisson
