'''
This module implements differential operators on Cartesian grids 

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_vector_laplace
   make_tensor_divergence
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>   
'''

from typing import Callable, Tuple, Any

import numba as nb
import numpy as np
from scipy import ndimage, sparse

from .common import (make_laplace_from_matrix, make_general_poisson_solver,
                     PARALLELIZATION_THRESHOLD_2D, PARALLELIZATION_THRESHOLD_3D)
from ..boundaries import Boundaries
from ..cartesian import CartesianGridBase
from ...tools.numba import jit_allocate_out
from ...tools.docstrings import fill_in_docstring



@fill_in_docstring
def _get_laplace_matrix_1d(bcs) -> Tuple[Any, Any]:
    """ get sparse matrix for laplace operator on a 1d Cartesian grid
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """    
    dim_x = bcs.grid.shape[0]
    matrix = sparse.dok_matrix((dim_x, dim_x))
    vector = sparse.dok_matrix((dim_x, 1))
    
    for i in range(dim_x):
        matrix[i, i] += -2
        
        if i == 0:
            const, entries = bcs[0].get_data((-1,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i - 1] += 1
            
        if i == dim_x - 1:
            const, entries = bcs[0].get_data((dim_x,))
            vector[i] += const
            for k, v in entries.items():
                matrix[i, k] += v
        else:
            matrix[i, i + 1] += 1

    matrix *= bcs.grid.discretization[0]**-2
    vector *= bcs.grid.discretization[0]**-2
            
    return matrix, vector

    

@fill_in_docstring
def _get_laplace_matrix_2d(bcs) -> Tuple[Any, Any]:
    """ get sparse matrix for laplace operator on a 2d Cartesian grid
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """        
    dim_x, dim_y = bcs.grid.shape
    matrix = sparse.dok_matrix((dim_x * dim_y, dim_x * dim_y))
    vector = sparse.dok_matrix((dim_x * dim_y, 1))
    
    bc_x, bc_y = bcs
    scale_x, scale_y = bcs.grid.discretization**-2
    
    def i(x, y):
        """ helper function for flattening the index
        
        This is equivalent to np.ravel_multi_index((x, y), (dim_x, dim_y))
        """
        return x * dim_y + y
        
    # set diagonal elements, i.e., the central value in the kernel
    matrix.setdiag(-2 * (scale_x + scale_y))
        
    for x in range(dim_x):
        for y in range(dim_y):
            # handle x-direction
            if x == 0:
                const, entries = bc_x.get_data((-1, y))
                vector[i(x, y)] += const * scale_x
                for k, v in entries.items():
                    matrix[i(x, y), i(k, y)] += v * scale_x
            else:
                matrix[i(x, y), i(x - 1, y)] += scale_x

            if x == dim_x - 1:
                const, entries = bc_x.get_data((dim_x, y))
                vector[i(x, y)] += const * scale_x
                for k, v in entries.items():
                    matrix[i(x, y), i(k, y)] += v * scale_x
            else:
                matrix[i(x, y), i(x + 1, y)] += scale_x  
                
            # handle y-direction
            if y == 0:
                const, entries = bc_y.get_data((x, -1))
                vector[i(x, y)] += const * scale_y
                for k, v in entries.items():
                    matrix[i(x, y), i(x, k)] += v * scale_y
            else:
                matrix[i(x, y), i(x, y - 1)] += scale_y

            if y == dim_y - 1:
                const, entries = bc_y.get_data((x, dim_y))
                vector[i(x, y)] += const * scale_y
                for k, v in entries.items():
                    matrix[i(x, y), i(x, k)] += v * scale_y
            else:
                matrix[i(x, y), i(x, y + 1)] += scale_y

    return matrix, vector



@fill_in_docstring
def _get_laplace_matrix(bcs: Boundaries):
    """ get sparse matrix for laplace operator on a 1d Cartesian grid
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        tuple: A sparse matrix and a sparse vector that can be used to evaluate
        the discretized laplacian
    """    
    dim = bcs.grid.dim
    bcs.check_value_rank(0)
    
    if dim == 1:
        result = _get_laplace_matrix_1d(bcs)
    elif dim == 2:
        result = _get_laplace_matrix_2d(bcs)
    else:
        raise NotImplementedError('Numba laplace operator not implemented '
                                  f'for {dim:d} dimensions')
                                      
    return result



def _make_derivative(bcs: Boundaries, axis: int = 0, method: str = 'central') \
        -> Callable:
    """ make a derivative operator along a single axis using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        axis (int):
            The axis along which the derivative will be taken
        method (str):
            The method for calculating the derivative. Possible values are
            'central', 'forward', and 'backward'.
        
    Returns:
        A function that can be applied to an array of values. The result will be
        an array of the same shape containing the actual derivatives at the grid
        points.
    """
    if method not in {'central', 'forward', 'backward'}:
        raise ValueError(f'Unknown derivative type `{method}`')
    
    shape = bcs.grid.shape
    dim = len(shape)
    dx = bcs.grid.discretization[axis]
    region = bcs[axis].make_region_evaluator()
        
    if dim == 1:
        @jit_allocate_out(out_shape=shape)
        def diff(arr, out=None):
            """ calculate derivative of 1d array `arr` """
            for i in range(shape[0]):
                arr_l, arr_m, arr_h = region(arr, (i,))
                if method == 'central':
                    out[i] = (arr_h - arr_l) * 0.5 / dx
                elif method == 'forward':
                    out[i] = (arr_h - arr_m) / dx
                elif method == 'backward':
                    out[i] = (arr_m - arr_l) / dx
                    
            return out       
        
    elif dim == 2:
        @jit_allocate_out(out_shape=shape)
        def diff(arr, out=None):
            """ calculate derivative of 2d array `arr` """
            for i in range(shape[0]):
                for j in range(shape[1]):
                    arr_l, arr_m, arr_h = region(arr, (i, j))
                    if method == 'central':
                        out[i, j] = (arr_h - arr_l) * 0.5 / dx
                    elif method == 'forward':
                        out[i, j] = (arr_h - arr_m) / dx
                    elif method == 'backward':
                        out[i, j] = (arr_m - arr_l) / dx
                    
            return out       
        
    elif dim == 3:
        @jit_allocate_out(out_shape=shape)
        def diff(arr, out=None):
            """ calculate derivative of 3d array `arr` """
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        arr_l, arr_m, arr_h = region(arr, (i, j, k))
                        if method == 'central':
                            out[i, j, k] = (arr_h - arr_l) * 0.5 / dx
                        elif method == 'forward':
                            out[i, j, k] = (arr_h - arr_m) / dx
                        elif method == 'backward':
                            out[i, j, k] = (arr_m - arr_l) / dx
                    
            return out     
        
    else:
        raise NotImplementedError('Numba derivative operator not implemented '
                                  f'for {dim:d} dimensions')
        
    return diff  # type: ignore    



@fill_in_docstring
def _make_laplace_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a laplace operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values
    """
    scaling = bcs._uniform_discretization**-2
    args = bcs._scipy_border_mode
    
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        return ndimage.laplace(scaling * arr, output=out, **args)
        
    return laplace


# def _make_laplace_fft_nd(shape, boundaries, dx=1):
#     """ make a laplace operator using the fft module """
#     
#     if not boundaries.periodic:
#         raise ValueError('Boundary needs to be periodic')
#     
#     dim = len(shape)
#     dx = np.broadcast_to(dx, (dim,))
#     ks = [np.fft.fftfreq(n, d) for n, d in zip(shape, dx)]
#     k2s = np.meshgrid(*ks, indexing='ij')**2
#     
#     def laplace(arr, out=None):
#         """ apply laplace operator to array `arr` """
#         arr_k = np.fft.rfftn(arr, s=shape)
#         return np.fft.irfftn(arr_k * k2s, s=shape)
#         
#     return laplace

        
    
@fill_in_docstring
def _make_laplace_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d laplace operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x = bcs.grid.shape[0]
    scale = bcs.grid.discretization[0]**-2
    region_x = bcs[0].make_region_evaluator()
    
    @jit_allocate_out
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        for i in range(dim_x):
            valm, val, valp = region_x(arr, (i,))
            out[i] = (valm - 2 * val + valp) * scale

        return out        
        
    return laplace  # type: ignore

    
    
@fill_in_docstring
def _make_laplace_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d laplace operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y = bcs.grid.shape
    scale_x, scale_y = bcs.grid.discretization**-2
    
    region_x = bcs[0].make_region_evaluator()
    region_y = bcs[1].make_region_evaluator()
    
    # use parallel processing for large enough arrays 
    parallel = (dim_x * dim_y >= PARALLELIZATION_THRESHOLD_2D**2)
    
    @jit_allocate_out(parallel=parallel)
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        for i in nb.prange(dim_x):

            # left-most part
            j = 0
            arr_x_l, arr_c, arr_x_h = region_x(arr, (i, j))
            arr_y_l, _, arr_y_h = region_y(arr, (i, j))
            lap_x = (arr_x_l - 2 * arr_c + arr_x_h) * scale_x
            lap_y = (arr_y_l - 2 * arr_c + arr_y_h) * scale_y
            out[i, j] = lap_x + lap_y
            
            if dim_y == 1:
                continue  # deal with singular y dimension
 
            # middle part without boundary conditions in y-direction
            for j in range(1, dim_y - 1):
                arr_x_l, arr_c, arr_x_h = region_x(arr, (i, j))
                arr_y_l = arr[i, j - 1]
                arr_y_h = arr[i, j + 1]
                 
                lap_x = (arr_x_l - 2 * arr_c + arr_x_h) * scale_x
                lap_y = (arr_y_l - 2 * arr_c + arr_y_h) * scale_y
                out[i, j] = lap_x + lap_y
         
            # right-most part
            j = dim_y - 1
            arr_x_l, arr_c, arr_x_h = region_x(arr, (i, j))
            arr_y_l, _, arr_y_h = region_y(arr, (i, j))
            lap_x = (arr_x_l - 2 * arr_c + arr_x_h) * scale_x
            lap_y = (arr_y_l - 2 * arr_c + arr_y_h) * scale_y
            out[i, j] = lap_x + lap_y

        return out        
        
    return laplace  # type: ignore
    


@fill_in_docstring
def _make_laplace_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d laplace operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = bcs.grid.shape 
    scale_x, scale_y, scale_z = bcs.grid.discretization**-2
    
    region_x = bcs[0].make_region_evaluator()
    region_y = bcs[1].make_region_evaluator()
    region_z = bcs[2].make_region_evaluator()
    
    # use parallel processing for large enough arrays 
    parallel = (dim_x * dim_y * dim_z >= PARALLELIZATION_THRESHOLD_3D**3)
    
    @jit_allocate_out(parallel=parallel)
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        for i in nb.prange(dim_x):
            for j in range(dim_y):
                for k in range(dim_z):

                    arr_x_l, arr_c, arr_x_h = region_x(arr, (i, j, k))
                    arr_y_l, _, arr_y_h = region_y(arr, (i, j, k))
                    arr_z_l, _, arr_z_h = region_z(arr, (i, j, k))
                    
                    lap_x = (arr_x_l - 2 * arr_c + arr_x_h) * scale_x
                    lap_y = (arr_y_l - 2 * arr_c + arr_y_h) * scale_y
                    lap_z = (arr_z_l - 2 * arr_c + arr_z_h) * scale_z
                    out[i, j, k] = lap_x + lap_y + lap_z

        return out        
        
    return laplace  # type: ignore



@fill_in_docstring
def make_laplace(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a laplace operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str): Method used for calculating the laplace operator.
            If method='auto', a suitable method is chosen automatically.
        
    Returns:
        A function that can be applied to an array of values
    """
    dim = bcs.grid.dim
    bcs.check_value_rank(0)
    
    if method == 'auto':
        # choose the fastest available Laplace operator
        if 1 <= dim <= 3:
            method = 'numba'
        else:
            method = 'scipy'
            
    if method == 'numba':
        if dim == 1:
            laplace = _make_laplace_numba_1d(bcs)
        elif dim == 2:
            laplace = _make_laplace_numba_2d(bcs)
        elif dim == 3:
            laplace = _make_laplace_numba_3d(bcs)
        else:
            raise NotImplementedError('Numba laplace operator not implemented '
                                      f'for {dim:d} dimensions')
                          
    elif method == 'matrix':
        laplace = make_laplace_from_matrix(*_get_laplace_matrix(bcs))
                                      
    elif method == 'scipy':
        laplace = _make_laplace_scipy_nd(bcs)
        
    else:
        raise ValueError(f'Method `{method}` is not defined')
        
    return laplace


# register operators with the grid class
CartesianGridBase.register_operator('laplace', make_laplace,
                                    rank_in=0, rank_out=0)



@fill_in_docstring
def _make_gradient_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a gradient operator using the scipy module
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values
    """    
    scaling = 0.5 / bcs._uniform_discretization
    dim = bcs.grid.dim
    shape_out = (dim,) + bcs.grid.shape 
    args = bcs._scipy_border_mode
    
    def gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        if out is None:
            out = np.empty(shape_out)
        for i in range(dim):
            out[i] = ndimage.convolve1d(arr, [1, 0, -1], axis=i, **args) 
        return out * scaling
        
    return gradient



@fill_in_docstring
def _make_gradient_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:    
        A function that can be applied to an array of values
    """
    dim_x = bcs.grid.shape[0]
    scale = 0.5 / bcs.grid.discretization[0]
    region_x = bcs[0].make_region_evaluator()
    
    @jit_allocate_out(out_shape=(1, dim_x))
    def gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in range(dim_x):
            valm, _, valp = region_x(arr, (i,))
            out[0, i] = (valp - valm) * scale
                
        return out        
        
    return gradient  # type: ignore



@fill_in_docstring
def _make_gradient_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y = bcs.grid.shape
    scale_x, scale_y = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].make_region_evaluator()
    region_y = bcs[1].make_region_evaluator()

    # use parallel processing for large enough arrays 
    parallel = (dim_x * dim_y >= PARALLELIZATION_THRESHOLD_2D**2)
        
    @jit_allocate_out(parallel=parallel, out_shape=(2, dim_x, dim_y))
    def gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in nb.prange(dim_x):
            for j in range(dim_y):
                arr_x_l, _, arr_x_h = region_x(arr, (i, j))
                arr_y_l, _, arr_y_h = region_y(arr, (i, j))
                    
                out[0, i, j] = (arr_x_h - arr_x_l) * scale_x
                out[1, i, j] = (arr_y_h - arr_y_l) * scale_y
                
        return out        
        
    return gradient  # type: ignore



@fill_in_docstring
def _make_gradient_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y, dim_z = bcs.grid.shape
    scale_x, scale_y, scale_z = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].make_region_evaluator()
    region_y = bcs[1].make_region_evaluator()
    region_z = bcs[2].make_region_evaluator()
    
    # use parallel processing for large enough arrays 
    parallel = (dim_x * dim_y * dim_z >= PARALLELIZATION_THRESHOLD_3D**3)
    
    @jit_allocate_out(parallel=parallel, out_shape=(3, dim_x, dim_y, dim_z))
    def gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in nb.prange(dim_x):
            for j in range(dim_y):
                for k in range(dim_z):
                    arr_x_l, _, arr_x_h = region_x(arr, (i, j, k))
                    arr_y_l, _, arr_y_h = region_y(arr, (i, j, k))
                    arr_z_l, _, arr_z_h = region_z(arr, (i, j, k))

                    out[0, i, j, k] = (arr_x_h - arr_x_l) * scale_x
                    out[1, i, j, k] = (arr_y_h - arr_y_l) * scale_y
                    out[2, i, j, k] = (arr_z_h - arr_z_l) * scale_z
                
        return out        
        
    return gradient  # type: ignore



@fill_in_docstring
def make_gradient(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a gradient operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str): Method used for calculating the gradient operator.
            If method='auto', a suitable method is chosen automatically.
        
    Returns:
        A function that can be applied to an array of values
    """
    dim = bcs.grid.dim
    bcs.check_value_rank(0)
    
    if method == 'auto':
        # choose the fastest available gradient operator
        if 1 <= dim <= 3:
            method = 'numba'
        else:
            method = 'scipy'

    if method == 'numba':
        if dim == 1:
            gradient = _make_gradient_numba_1d(bcs)
        elif dim == 2:
            gradient = _make_gradient_numba_2d(bcs)
        elif dim == 3:
            gradient = _make_gradient_numba_3d(bcs)
        else:
            raise NotImplementedError('Numba gradient operator not '
                                      f'implemented for {dim}')
                                      
    elif method == 'scipy':
        gradient = _make_gradient_scipy_nd(bcs)
        
    else:
        raise ValueError(f'Method `{method}` is not defined')
        
    return gradient


# register operators with the grid class
CartesianGridBase.register_operator('gradient', make_gradient,
                                    rank_in=0, rank_out=1)



@fill_in_docstring
def _make_divergence_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a divergence operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values
    """
    shape = bcs.grid.shape
    scaling = 0.5 / bcs._uniform_discretization
    args = bcs._scipy_border_mode
    
    def divergence(arr, out=None):
        """ apply divergence operator to array `arr` """
        assert arr.shape[0] == len(shape) and len(arr.shape) == len(shape) + 1
        # need to initialize with zeros since data is added later
        if out is None:
            out = np.zeros(arr.shape[1:])
        else:
            out[:] = 0
            
        for i in range(len(shape)):
            out += ndimage.convolve1d(arr[i], [1, 0, -1], axis=i, **args) 
        return out * scaling
        
    return divergence



@fill_in_docstring
def _make_divergence_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d divergence operator using numba compilation
    
    Args:
        dim (int): The number of support points for each axes
        boundaries (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        dx (float): The discretization
        
    Returns:    
        A function that can be applied to an array of values
    """
    dim_x = bcs.grid.shape[0]
    scale = 0.5 / bcs.grid.discretization[0]
    region_x = bcs[0].make_region_evaluator()
    
    @jit_allocate_out(out_shape=(dim_x,))
    def divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in range(dim_x):
            valm, _, valp = region_x(arr[0], (i,))
            out[i] = (valp - valm) * scale
                
        return out        
        
    return divergence  # type: ignore



@fill_in_docstring
def _make_divergence_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d divergence operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y = bcs.grid.shape
    scale_x, scale_y = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].make_region_evaluator()
    region_y = bcs[1].make_region_evaluator()

    # use parallel processing for large enough arrays 
    parallel = (dim_x * dim_y >= PARALLELIZATION_THRESHOLD_2D**2)
    
    @jit_allocate_out(parallel=parallel, out_shape=(dim_x, dim_y))
    def divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in nb.prange(dim_x):
            for j in range(dim_y):
                arr_x_l, _, arr_x_h = region_x(arr[0], (i, j))
                arr_y_l, _, arr_y_h = region_y(arr[1], (i, j))
                    
                d_x = (arr_x_h - arr_x_l) * scale_x
                d_y = (arr_y_h - arr_y_l) * scale_y
                out[i, j] = d_x + d_y
                
        return out        
        
    return divergence  # type: ignore



@fill_in_docstring
def _make_divergence_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d divergence operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y, dim_z = bcs.grid.shape
    scale_x, scale_y, scale_z = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].make_region_evaluator()
    region_y = bcs[1].make_region_evaluator()
    region_z = bcs[2].make_region_evaluator()
    
    # use parallel processing for large enough arrays 
    parallel = (dim_x * dim_y * dim_z >= PARALLELIZATION_THRESHOLD_3D**3)
    
    @jit_allocate_out(parallel=parallel, out_shape=(dim_x, dim_y, dim_z))
    def divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in nb.prange(dim_x):
            for j in range(dim_y):
                for k in range(dim_z):
                    arr_x_l, _, arr_x_h = region_x(arr[0], (i, j, k))
                    arr_y_l, _, arr_y_h = region_y(arr[1], (i, j, k))
                    arr_z_l, _, arr_z_h = region_z(arr[2], (i, j, k))
                        
                    d_x = (arr_x_h - arr_x_l) * scale_x
                    d_y = (arr_y_h - arr_y_l) * scale_y
                    d_z = (arr_z_h - arr_z_l) * scale_z
                    out[i, j, k] = d_x + d_y + d_z
                
        return out        
        
    return divergence  # type: ignore



@fill_in_docstring
def make_divergence(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a divergence operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str): Method used for calculating the divergence operator.
            If method='auto', a suitable method is chosen automatically.
        
    Returns:
        A function that can be applied to an array of values
    """    
    dim = bcs.grid.dim
    bcs.check_value_rank(0)
    
    if method == 'auto':
        # choose the fastest available divergence operator
        if 1 <= dim <= 3:
            method = 'numba'
        else:
            method = 'scipy'

    if method == 'numba':
        if dim == 1:
            divergence = _make_divergence_numba_1d(bcs)
        elif dim == 2:
            divergence = _make_divergence_numba_2d(bcs)
        elif dim == 3:
            divergence = _make_divergence_numba_3d(bcs)
        else:
            raise NotImplementedError('Numba divergence operator not '
                                      f'implemented for {dim}')
                                      
    elif method == 'scipy':
        divergence = _make_divergence_scipy_nd(bcs)
        
    else:
        raise ValueError(f'Method `{method}` is not defined')
        
    return divergence


# register operators with the grid class
CartesianGridBase.register_operator('divergence', make_divergence,
                                    rank_in=1, rank_out=0)



@fill_in_docstring
def _make_vector_gradient_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a vector gradient operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values
    """
    scaling = 0.5 / bcs._uniform_discretization
    args = bcs._scipy_border_mode
    dim = bcs.grid.dim
    shape_out = (dim, dim) + bcs.grid.shape 
    
    def vector_gradient(arr, out=None):
        """ apply vector gradient operator to array `arr` """
        if out is None:
            out = np.empty(shape_out)
        for i in range(dim):
            for j in range(dim):
                out[i, j] = ndimage.convolve1d(arr[j], [1, 0, -1],
                                               axis=i, **args) 
        return out * scaling
        
    return vector_gradient



@fill_in_docstring
def _make_vector_gradient_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d vector gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:    
        A function that can be applied to an array of values
    """
    gradient = _make_gradient_numba_1d(bcs.extract_component(0))
        
    @jit_allocate_out(out_shape=(1, 1) + bcs.grid.shape)
    def vector_gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        gradient(arr[0], out=out[0])
        return out
        
    return vector_gradient  # type: ignore



@fill_in_docstring
def _make_vector_gradient_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d vector gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    gradient_x = _make_gradient_numba_2d(bcs.extract_component(0))
    gradient_y = _make_gradient_numba_2d(bcs.extract_component(1))
        
    @jit_allocate_out(out_shape=(2, 2) + bcs.grid.shape)
    def vector_gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        gradient_x(arr[0], out=out[:, 0])
        gradient_y(arr[1], out=out[:, 1])
        return out        
        
    return vector_gradient  # type: ignore



@fill_in_docstring
def _make_vector_gradient_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d vector gradient operator using numba compilation
    
    Args:
        shape (tuple): The number of support points for each axes
        boundaries (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        dx (float or list): The discretizations
        
    Returns:
        A function that can be applied to an array of values    
    """
    gradient_x = _make_gradient_numba_3d(bcs.extract_component(0))
    gradient_y = _make_gradient_numba_3d(bcs.extract_component(1))
    gradient_z = _make_gradient_numba_3d(bcs.extract_component(2))
        
    @jit_allocate_out(out_shape=(3, 3) + bcs.grid.shape)
    def vector_gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        gradient_x(arr[0], out=out[:, 0])
        gradient_y(arr[1], out=out[:, 1])
        gradient_z(arr[2], out=out[:, 2])
        return out        
        
    return vector_gradient  # type: ignore



@fill_in_docstring
def make_vector_gradient(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a vector gradient operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str): Method used for calculating the vector gradient operator.
            If method='auto', a suitable method is chosen automatically
        
    Returns:
        A function that can be applied to an array of values
    """
    dim = bcs.grid.dim
    bcs.check_value_rank(1)
    
    # choose the fastest available vector gradient operator
    if method == 'auto':
        if 1 <= dim <= 3:
            method = 'numba'
        else:
            method = 'scipy'

    if method == 'numba':
        if dim == 1:
            gradient = _make_vector_gradient_numba_1d(bcs)
        elif dim == 2:
            gradient = _make_vector_gradient_numba_2d(bcs)
        elif dim == 3:
            gradient = _make_vector_gradient_numba_3d(bcs)
        else:
            raise NotImplementedError('Numba vector gradient operator not '
                                      f'implemented for {dim}')
                                      
    elif method == 'scipy':
        gradient = _make_vector_gradient_scipy_nd(bcs)
    else:
        raise ValueError(f'Method `{method}` is not defined')
        
    return gradient


# register operators with the grid class
CartesianGridBase.register_operator('vector_gradient', make_vector_gradient,
                                    rank_in=1, rank_out=2)



@fill_in_docstring
def _make_vector_laplace_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a vector Laplacian using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values
    """
    scaling = bcs._uniform_discretization**-2
    args = bcs._scipy_border_mode
    dim = bcs.grid.dim
    shape_out = (dim,) + bcs.grid.shape
    
    def vector_laplace(arr, out=None):
        """ apply vector Laplacian operator to array `arr` """
        if out is None:
            out = np.empty(shape_out)
        for i in range(dim):
            ndimage.laplace(arr[i], output=out[i], **args)
        return out * scaling
        
    return vector_laplace



@fill_in_docstring
def _make_vector_laplace_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d vector Laplacian using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:    
        A function that can be applied to an array of values
    """
    laplace = _make_laplace_numba_1d(bcs.extract_component(0))
        
    @jit_allocate_out(out_shape=(1,) + bcs.grid.shape)
    def vector_laplace(arr, out=None):
        """ apply vector Laplacian to array `arr` """
        laplace(arr[0], out=out[0])
        return out
        
    return vector_laplace  # type: ignore



@fill_in_docstring
def _make_vector_laplace_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d vector Laplacian using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    laplace_x = _make_laplace_numba_2d(bcs.extract_component(0))
    laplace_y = _make_laplace_numba_2d(bcs.extract_component(1))
        
    @jit_allocate_out(out_shape=(2,) + bcs.grid.shape)
    def vector_laplace(arr, out=None):
        """ apply vector Laplacian  to array `arr` """
        laplace_x(arr[0], out=out[0])
        laplace_y(arr[1], out=out[1])
        return out        
        
    return vector_laplace  # type: ignore



@fill_in_docstring
def _make_vector_laplace_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d vector Laplacian using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    laplace_x = _make_laplace_numba_3d(bcs.extract_component(0))
    laplace_y = _make_laplace_numba_3d(bcs.extract_component(1))
    laplace_z = _make_laplace_numba_3d(bcs.extract_component(2))
        
    @jit_allocate_out(out_shape=(3,) + bcs.grid.shape)
    def vector_laplace(arr, out=None):
        """ apply vector Laplacian to array `arr` """
        laplace_x(arr[0], out=out[0])
        laplace_y(arr[1], out=out[1])
        laplace_z(arr[2], out=out[2])
        return out        
        
    return vector_laplace  # type: ignore



@fill_in_docstring
def make_vector_laplace(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a vector Laplacian on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str): Method used for calculating the vector laplace operator.
            If method='auto', a suitable method is chosen automatically.
        
    Returns:
        A function that can be applied to an array of values
    """
    dim = bcs.grid.dim
    bcs.check_value_rank(1)
    
    # choose the fastest available vector gradient operator
    if method == 'auto':
        if 1 <= dim <= 3:
            method = 'numba'
        else:
            method = 'scipy'

    if method == 'numba':
        if dim == 1:
            gradient = _make_vector_laplace_numba_1d(bcs)
        elif dim == 2:
            gradient = _make_vector_laplace_numba_2d(bcs)
        elif dim == 3:
            gradient = _make_vector_laplace_numba_3d(bcs)
        else:
            raise NotImplementedError('Numba vector gradient operator not '
                                      f'implemented for {dim}')
                                      
    elif method == 'scipy':
        gradient = _make_vector_laplace_scipy_nd(bcs)
    else:
        raise ValueError(f'Method `{method}` is not defined')
        
    return gradient


# register operators with the grid class
CartesianGridBase.register_operator('vector_laplace', make_vector_laplace,
                                    rank_in=1, rank_out=1)



@fill_in_docstring
def _make_tensor_divergence_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a tensor divergence operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values
    """
    scaling = 0.5 / bcs._uniform_discretization
    args = bcs._scipy_border_mode
    dim = bcs.grid.dim
    shape_out = (dim,) + bcs.grid.shape
    
    def tensor_divergence(arr, out=None):
        """ apply tensor divergence operator to array `arr` """
        # need to initialize with zeros since data is added later
        if out is None:
            out = np.zeros(shape_out)
        else:
            out[:] = 0
            
        for i in range(dim):
            for j in range(dim):
                out[i] += ndimage.convolve1d(arr[i, j], [1, 0, -1],
                                             axis=j, **args) 
        return out * scaling
        
    return tensor_divergence



@fill_in_docstring
def _make_tensor_divergence_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d tensor divergence  operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:    
        A function that can be applied to an array of values
    """
    divergence = _make_divergence_numba_1d(bcs.extract_component(0))
        
    @jit_allocate_out(out_shape=(1,) + bcs.grid.shape)
    def tensor_divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        divergence(arr[0], out=out[0])
        return out
        
    return tensor_divergence  # type: ignore



@fill_in_docstring
def _make_tensor_divergence_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d tensor divergence  operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    divergence_x = _make_divergence_numba_2d(bcs.extract_component(0))
    divergence_y = _make_divergence_numba_2d(bcs.extract_component(1))
        
    @jit_allocate_out(out_shape=(2,) + bcs.grid.shape)
    def tensor_divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        divergence_x(arr[0], out=out[0])
        divergence_y(arr[1], out=out[1])
        return out        
        
    return tensor_divergence  # type: ignore



@fill_in_docstring
def _make_tensor_divergence_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d tensor divergence  operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        
    Returns:
        A function that can be applied to an array of values    
    """
    divergence_x = _make_divergence_numba_3d(bcs.extract_component(0))
    divergence_y = _make_divergence_numba_3d(bcs.extract_component(1))
    divergence_z = _make_divergence_numba_3d(bcs.extract_component(2))
        
    @jit_allocate_out(out_shape=(3,) + bcs.grid.shape)
    def tensor_divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        divergence_x(arr[0], out=out[0])
        divergence_y(arr[1], out=out[1])
        divergence_z(arr[2], out=out[2])
        return out        
        
    return tensor_divergence  # type: ignore



@fill_in_docstring
def make_tensor_divergence(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a tensor divergence operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str): Method used for calculating the tensor divergence
            operator. If method='auto', a suitable method is chosen
            automatically.
        
    Returns:
        A function that can be applied to an array of values
    """
    dim = bcs.grid.dim
    bcs.check_value_rank(1)
    
    # choose the fastest available tensor divergence operator
    if method == 'auto':
        if 1 <= dim <= 3:
            method = 'numba'
        else:
            method = 'scipy'

    if method == 'numba':
        if dim == 1:
            func = _make_tensor_divergence_numba_1d(bcs)
        elif dim == 2:
            func = _make_tensor_divergence_numba_2d(bcs)
        elif dim == 3:
            func = _make_tensor_divergence_numba_3d(bcs)
        else:
            raise NotImplementedError('Numba tensor divergence operator not '
                                      f'implemented for {dim}')
                                      
    elif method == 'scipy':
        func = _make_tensor_divergence_scipy_nd(bcs)
    else:
        raise ValueError(f'Method `{method}` is not defined')
        
    return func


# register operators with the grid class
CartesianGridBase.register_operator('tensor_divergence', make_tensor_divergence,
                                    rank_in=2, rank_out=1)



@fill_in_docstring
def make_poisson_solver(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a operator that solves Poisson's equation

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            {ARG_BOUNDARIES_INSTANCE}
        method (str): Method used for calculating the tensor divergence
            operator. If method='auto', a suitable method is chosen
            automatically.
        
    Returns:
        A function that can be applied to an array of values
    """
    matrix, vector = _get_laplace_matrix(bcs)
    return make_general_poisson_solver(matrix, vector, method)


# register operators with the grid class
CartesianGridBase.register_operator('poisson_solver', make_poisson_solver,
                                    rank_in=0, rank_out=0)



__all__ = ["make_laplace", "make_gradient", "make_divergence",
           "make_vector_gradient", "make_vector_laplace",
           "make_tensor_divergence", "make_poisson_solver"]
    