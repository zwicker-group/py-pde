r"""
This module implements differential operators on polar grids 

.. autosummary::
   :nosignatures:

   make_laplace
   make_gradient
   make_divergence
   make_vector_gradient
   make_tensor_divergence
   make_operator
   
   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>


.. The following contains text parts that are used multiple times below:

.. |Description_polar| replace:: 
  This function assumes polar symmetry of the grid, so that fields only
  depend on the radial coordinate `r`. The radial discretization is defined as
  :math:`r_i = r_\mathrm{min} + (i + \frac12) \Delta r` for
  :math:`i=0, \ldots, N_r-1`,  where :math:`r_\mathrm{min}` is the radius of
  the inner boundary, which is zero by default. Note that the radius of the 
  outer boundary is given by
  :math:`r_\mathrm{max} = r_\mathrm{min} + N_r \Delta r`.
"""

from typing import Callable

from .. import PolarGrid
from ..boundaries import Boundaries
from ...tools.numba import jit_allocate_out



def make_laplace(bcs: Boundaries) -> Callable:
    """ make a discretized laplace operator for a polar grid
    
    |Description_polar|

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)
    
    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    dr_2 = 1 / dr**2

    # prepare boundary values
    value_lower_bc = bcs[0].low.get_virtual_point_evaluator()
    value_upper_bc = bcs[0].high.get_virtual_point_evaluator()    
    
    @jit_allocate_out(out_shape=(dim_r,))
    def laplace(arr, out=None):
        """ apply laplace operator to array `arr` """
        i = 0
        if r_min == 0:
            out[i] = 2 * (arr[i + 1] - arr[i]) * dr_2
        else:
            arr_r_l = value_lower_bc(arr, (i,))
            out[i] = ((arr[i + 1] - 2 * arr[i] + arr_r_l) * dr_2 +
                      (arr[i + 1] - arr_r_l) / (2 * rs[i] * dr))
        
        for i in range(1, dim_r - 1):  # iterate inner radial points
            out[i] = ((arr[i + 1] - 2 * arr[i] + arr[i - 1]) * dr_2 +
                      (arr[i + 1] - arr[i - 1]) / (2 * rs[i] * dr))
            
        # express boundary condition at outer side
        i = dim_r - 1
        arr_r_h = value_upper_bc(arr, (i,))
        out[i] = ((arr_r_h - 2 * arr[i] + arr[i - 1]) * dr_2 +
                  (arr_r_h - arr[i - 1]) / (2 * rs[i] * dr))
        return out      
          
    return laplace  # type: ignore



def make_gradient(bcs: Boundaries) -> Callable:
    """ make a discretized gradient operator for a polar grid
    
    |Description_polar|

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    dr = bcs.grid.discretization[0]
    scale_r = 1 / (2 * dr)
    
    # prepare boundary values
    boundary = bcs[0]
    value_lower_bc = boundary.low.get_virtual_point_evaluator()
    value_upper_bc = boundary.high.get_virtual_point_evaluator()
    
    @jit_allocate_out(out_shape=(2, dim_r))
    def gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        # no-flux at the origin 
        i = 0
        if r_min == 0:
            out[0, i] = (arr[1] - arr[0]) * scale_r
        else:
            arr_r_l = value_lower_bc(arr, (i,))
            out[0, i] = (arr[1] - arr_r_l) * scale_r            
        out[1, i] = 0  # no angular dependence by definition
        
        for i in range(1, dim_r - 1):  # iterate inner radial points
            out[0, i] = (arr[i + 1] - arr[i - 1]) * scale_r
            out[1, i] = 0  # no angular dependence by definition

        i = dim_r - 1
        arr_r_h = value_upper_bc(arr, (i,))
        out[0, i] = (arr_r_h - arr[i - 1]) * scale_r
        out[1, i] = 0  # no angular dependence by definition
        
        return out
        
    return gradient  # type: ignore



def make_divergence(bcs: Boundaries) -> Callable:
    """ make a discretized divergence operator for a polar grid
    
    |Description_polar|

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(0)

    # calculate preliminary quantities
    dim_r = bcs.grid.shape[0]
    dr = bcs.grid.discretization[0]
    rs = bcs.grid.axes_coords[0]
    r_min, _ = bcs.grid.axes_bounds[0]
    scale_r = 1 / (2 * dr)
    
    # prepare boundary values
    boundary = bcs[0]
    value_lower_bc = boundary.low.get_virtual_point_evaluator()
    value_upper_bc = boundary.high.get_virtual_point_evaluator()

    if r_min == 0:
        @jit_allocate_out(out_shape=(dim_r,))
        def divergence(arr, out=None):
            """ apply divergence operator to array `arr` """            
            # inner radial boundary condition
            i = 0
            out[i] = (arr[0, 1] + 3 * arr[0, 0]) * scale_r
            
            for i in range(1, dim_r - 1):  # iterate radial points
                out[i] = ((arr[0, i + 1] - arr[0, i - 1]) * scale_r + 
                          (arr[0, i] / ((i + 0.5) * dr)))
                
            # outer radial boundary condition
            i = dim_r - 1
            arr_r_h = value_upper_bc(arr[0], (i,))
            out[i] = ((arr_r_h - arr[0, i - 1]) * scale_r + 
                      (arr[0, i] / ((i + 0.5) * dr)))
                
            return out
        
    else:  # r_min > 0
        @jit_allocate_out(out_shape=(dim_r,))
        def divergence(arr, out=None):
            """ apply divergence operator to array `arr` """            
            # inner radial boundary condition
            i = 0
            arr_r_l = value_lower_bc(arr[0], (i,))
            out[i] = (arr[0, i + 1] - arr_r_l) * scale_r + arr[0, i] / rs[i]
            
            for i in range(1, dim_r - 1):  # iterate radial points
                out[i] = ((arr[0, i + 1] - arr[0, i - 1]) * scale_r + 
                          arr[0, i] / rs[i])
                
            # outer radial boundary condition
            i = dim_r - 1
            arr_r_h = value_upper_bc(arr[0], (i,))
            out[i] = (arr_r_h - arr[0, i - 1]) * scale_r + arr[0, i] / rs[i]
                
            return out
        
    return divergence  # type: ignore

    
    
def make_vector_gradient(bcs: Boundaries) -> Callable:
    """ make a discretized vector gradient operator for a polar grid
    
    |Description_polar|

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(1)

    gradient_r = make_gradient(bcs.extract_component(0))
    gradient_phi = make_gradient(bcs.extract_component(1))
        
    @jit_allocate_out(out_shape=(2, 2) + bcs.grid.shape)
    def vector_gradient(arr, out=None):
        """ apply gradient operator to array `arr` """
        gradient_r(arr[0], out=out[:, 0])
        gradient_phi(arr[1], out=out[:, 1])
        return out    
        
    return vector_gradient  # type: ignore



def make_tensor_divergence(bcs: Boundaries) -> Callable:
    """ make a discretized tensor divergence operator for a polar grid
    
    |Description_polar|

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(bcs.grid, PolarGrid)
    bcs.check_value_rank(1)

    divergence_r = make_divergence(bcs.extract_component(0))
    divergence_phi = make_divergence(bcs.extract_component(1))
        
    @jit_allocate_out(out_shape=(2,) + bcs.grid.shape)
    def tensor_divergence(arr, out=None):
        """ apply gradient operator to array `arr` """
        divergence_r(arr[0], out=out[0])
        divergence_phi(arr[1], out=out[1])
        return out
        
    return tensor_divergence  # type: ignore



def make_operator(op: str, bcs: Boundaries) -> Callable:
    """ make a discretized operator for a polar grid
    
    |Description_polar|

    Args:
        op (str): Identifier for the operator. Some examples are 'laplace',
            'gradient', or 'divergence'.
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            |Arg_boundary_conditions|
        
    Returns:
        A function that takes the discretized data as an input and returns
        the data to which the operator `op` has been applied. This function
        optionally supports a second argument, which provides allocated
        memory for the output.
    """
    if op == 'laplace' or op == 'laplacian':
        return make_laplace(bcs)
    elif op == 'gradient':
        return make_gradient(bcs)
    elif op == 'divergence':
        return make_divergence(bcs)
    elif op == 'vector_gradient':
        return make_vector_gradient(bcs)
    elif op == 'tensor_divergence':
        return make_tensor_divergence(bcs)
    else:
        raise NotImplementedError(f'Operator `{op}` is not defined for '
                                  'polar grids')
        
        

    
