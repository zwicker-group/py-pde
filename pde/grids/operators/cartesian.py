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
   make_operator
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>   
'''

from typing import Callable

import numba as nb
import numpy as np
from scipy import ndimage

from . import PARALLELIZATION_THRESHOLD_2D, PARALLELIZATION_THRESHOLD_3D
from ..boundaries import Boundaries
from ...tools.numba import jit_allocate_out



def _make_laplace_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a laplace operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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

        
    
def _make_laplace_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d laplace operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x = bcs.grid.shape[0]
    scale = bcs.grid.discretization[0]**-2
    region_x = bcs[0].get_region_evaluator()
    
    @jit_allocate_out
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        for i in range(dim_x):
            valm, val, valp = region_x(arr, (i,))
            out[i] = (valm - 2 * val + valp) * scale

        return out        
        
    return laplace  # type: ignore

    
    
def _make_laplace_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d laplace operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y = bcs.grid.shape
    scale_x, scale_y = bcs.grid.discretization**-2
    
    region_x = bcs[0].get_region_evaluator()
    region_y = bcs[1].get_region_evaluator()
    
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
    


def _make_laplace_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d laplace operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values
    """
    dim_x, dim_y, dim_z = bcs.grid.shape 
    scale_x, scale_y, scale_z = bcs.grid.discretization**-2
    
    region_x = bcs[0].get_region_evaluator()
    region_y = bcs[1].get_region_evaluator()
    region_z = bcs[2].get_region_evaluator()
    
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



def make_laplace(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a laplace operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
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
                                      
    elif method == 'scipy':
        laplace = _make_laplace_scipy_nd(bcs)
        
    else:
        raise ValueError(f'Method `{method}` is not defined')
        
    return laplace



def _make_gradient_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a gradient operator using the scipy module
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_gradient_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:    
        A function that can be applied to an array of values
    """
    dim_x = bcs.grid.shape[0]
    scale = 0.5 / bcs.grid.discretization[0]
    region_x = bcs[0].get_region_evaluator()
    
    @jit_allocate_out(out_shape=(1, dim_x))
    def gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in range(dim_x):
            valm, _, valp = region_x(arr, (i,))
            out[0, i] = (valp - valm) * scale
                
        return out        
        
    return gradient  # type: ignore



def _make_gradient_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y = bcs.grid.shape
    scale_x, scale_y = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].get_region_evaluator()
    region_y = bcs[1].get_region_evaluator()

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



def _make_gradient_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y, dim_z = bcs.grid.shape
    scale_x, scale_y, scale_z = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].get_region_evaluator()
    region_y = bcs[1].get_region_evaluator()
    region_z = bcs[2].get_region_evaluator()
    
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



def make_gradient(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a gradient operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
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



def _make_divergence_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a divergence operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_divergence_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d divergence operator using numba compilation
    
    Args:
        dim (int): The number of support points for each axes
        boundaries (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        dx (float): The discretization
        
    Returns:    
        A function that can be applied to an array of values
    """
    dim_x = bcs.grid.shape[0]
    scale = 0.5 / bcs.grid.discretization[0]
    region_x = bcs[0].get_region_evaluator()
    
    @jit_allocate_out(out_shape=(dim_x,))
    def divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        for i in range(dim_x):
            valm, _, valp = region_x(arr[0], (i,))
            out[i] = (valp - valm) * scale
                
        return out        
        
    return divergence  # type: ignore



def _make_divergence_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d divergence operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y = bcs.grid.shape
    scale_x, scale_y = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].get_region_evaluator()
    region_y = bcs[1].get_region_evaluator()

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



def _make_divergence_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d divergence operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values    
    """
    dim_x, dim_y, dim_z = bcs.grid.shape
    scale_x, scale_y, scale_z = 0.5 / bcs.grid.discretization
    
    region_x = bcs[0].get_region_evaluator()
    region_y = bcs[1].get_region_evaluator()
    region_z = bcs[2].get_region_evaluator()
    
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



def make_divergence(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a divergence operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
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



def _make_vector_gradient_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a vector gradient operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_vector_gradient_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d vector gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_vector_gradient_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d vector gradient operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_vector_gradient_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d vector gradient operator using numba compilation
    
    Args:
        shape (tuple): The number of support points for each axes
        boundaries (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
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



def make_vector_gradient(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a vector gradient operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
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


def _make_vector_laplace_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a vector Laplacian using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_vector_laplace_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d vector Laplacian using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_vector_laplace_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d vector Laplacian using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_vector_laplace_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d vector Laplacian using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def make_vector_laplace(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a vector Laplacian on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
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



def _make_tensor_divergence_scipy_nd(bcs: Boundaries) -> Callable:
    """ make a tensor divergence operator using the scipy module
    
    This only supports uniform discretizations.
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_tensor_divergence_numba_1d(bcs: Boundaries) -> Callable:
    """ make a 1d tensor divergence  operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_tensor_divergence_numba_2d(bcs: Boundaries) -> Callable:
    """ make a 2d tensor divergence  operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def _make_tensor_divergence_numba_3d(bcs: Boundaries) -> Callable:
    """ make a 3d tensor divergence  operator using numba compilation
    
    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
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



def make_tensor_divergence(bcs: Boundaries, method: str = 'auto') -> Callable:
    """ make a tensor divergence operator on a Cartesian grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
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



# def rfftnfreq(shape: Tuple[int, ...], dx=1):
#     """ Return the Discrete Fourier Transform sample frequencies (for usage
#     with rfftn, irfftn).
#     
#     Args:
#         shape (tuple): the length of each axis
#         dx (float or sequence): the discretization along each axis. If a
#             single number is given, the same discretization is assumed along 
#             each axis
#             
#     Returns:
#         numpy.ndarray
#     """
#     dim = len(shape)
#     dx = np.broadcast_to(dx, (dim,))
#     
#     k2s = 0
#     for i in range(dim):
#         # get wave vector for axis i
#         if i == dim - 1:
#             k = np.fft.rfftfreq(shape[i], dx[i])
#         else:
#             k = np.fft.fftfreq(shape[i], dx[i])
#         # add the squared components to all present ones
#         k2s = np.add.outer(k2s, k**2)
#         
#     # get the magnitude at each position
#     return np.sqrt(k2s)
# 
# 
# 
# def _make_poisson_solver_scipy_nd(shape: Tuple[int, ...], dx) -> Callable:
#     """ make an operator that solves Poisson's equation on a Cartesian grid
#     using scipy.
#     
#     Args:
#         shape (tuple): The number of support points for each axes
#         dx (float or list): The discretizations
#         
#     Returns:
#         A function that can be applied to an array of values
#     """        
#     # prepare wave vector
#     k2s = rfftnfreq(shape, dx)**2
#         
#     # TODO: accelerate the FFT using the pyfftw package
#         
#     def poisson_solver(arr):
#         """ apply poisson solver to array `arr` """
#         # forward transform
#         arr = np.fft.rfftn(arr)
# 
#         # divide by squared wave vector
#         arr.flat[0] = 0  # remove zero mode
#         arr.flat[1:] /= k2s.flat[1:]
# 
#         # backwards transform
#         arr = np.fft.irfftn(arr, shape)
#         return arr
#     
#     return poisson_solver
# 
# 
# 
# def make_poisson_solver(shape: Tuple[int, ...], boundaries: Boundaries, dx=1,
#                         method: str='auto') -> Callable:
#     r""" make an operator that solves Poisson's equation on a Cartesian grid
#     
#     Denoting the current field by :math:`x`, we thus solve for :math:`y`,
#     defined by the equation 
# 
#     .. math::
#         \nabla^2 y(\boldsymbol r) = -x(\boldsymbol r)
#      
#         
#     Currently, this is only supported for periodic boundary conditions.
#     
#     Args:
#         shape (tuple): The number of support points for each axes
#         boundaries (:class:`~pde.grids.boundaries.axes.Boundaries`):
#             The boundary conditions. If the boundary conditions are not given
#             they are assumed to be all periodic.
#         dx (float or list): The discretizations
#         
#     Returns:
#         A function that can be applied to an array of values
#     """    
#     dim = len(shape)
#     if boundaries is not None:
#         boundaries.check_dimensions(dim=dim)
#         if not np.all(boundaries.periodic):
#             raise NotImplementedError("Solving Poisson's equation is only "
#                                       "implemented for periodic boundary "
#                                       "conditions")
# 
#     if method == 'auto':
#         method = 'scipy'
# 
#     if method == 'scipy':
#         poisson_solver = _make_poisson_solver_scipy_nd(shape, dx)
#     else:
#         raise ValueError(f'Method `{method}` is not defined')
#         
#     return poisson_solver



def make_operator(op: str, bcs: Boundaries, method: str = 'auto') -> Callable:
    """ return a discretized operator for a Cartesian grid
    
    Args:
        op (str): Identifier for the operator. Some examples are 'laplace',
            'gradient', or 'divergence'.
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        method (str): The method for implementing the operator
        
    Returns:
        A function that takes the discretized data as an input and returns
        the data to which the operator `op` has been applied. This function
        optionally supports a second argument, which provides allocated
        memory for the output.
    """
    if op == 'laplace' or op == 'laplacian':
        return make_laplace(bcs, method)
    elif op == 'gradient':
        return make_gradient(bcs, method)
    elif op == 'divergence':
        return make_divergence(bcs, method)
    elif op == 'vector_gradient':
        return make_vector_gradient(bcs, method) 
    elif op == 'vector_laplace' or op == 'vector_laplacian':
        return make_vector_laplace(bcs, method)
    elif op == 'tensor_divergence':
        return make_tensor_divergence(bcs, method)
#     elif op == 'poisson_solver':
#         return make_poisson_solver(shape, bc, dx, method)
    else:
        raise NotImplementedError(f'Operator `{op}` is not defined for '
                                  'Cartesian grids')
    


__all__ = ["make_laplace", "make_gradient", "make_divergence",
           "make_vector_gradient", "make_vector_laplace",
           "make_tensor_divergence", "make_operator"]
