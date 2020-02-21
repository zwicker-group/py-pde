"""
The Kardar–Parisi–Zhang (KPZ) equation describing the evolution of an interface

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import (Callable, Dict, Any, Counter, List)  # @UnusedImport

import numpy as np


from .base import PDEBase
from ..fields import ScalarField
from ..grids.boundaries.axes import BoundariesData
from ..tools.numba import nb, jit


        
class KuramotoSivashinskyPDE(PDEBase):
    r""" Represents the Kuramoto-Sivashinsky equation,
    
    .. math::
        \partial_t u = -\nu \nabla^4 u  - \nabla^2 u -
            \frac{1}{2} \left(\nabla h\right)^2  + \eta(\boldsymbol r, t)
        
    where :math:`u` is the height of the interface in Monge parameterization.
    The dynamics are governed by the parameters :math:`\nu` , while
    :math:`\eta` is Gaussian white noise, whose strength is controlled by the
    `noise` argument.
    """

    explicit_time_dependence = False
    

    def __init__(self, nu: float = 1,
                 noise: float = 0,
                 bc: BoundariesData = 'natural',
                 bc2: BoundariesData = 'natural'):
        r""" 
        Args:
            nu (float):
                Parameter :math:`\nu` for the strength of the fourth-order term
            noise (float):
                Strength of the (additive) noise term
            bc:
                The boundary conditions applied to the scalar field :math:`u`.
                The default value ('natural') imposes periodic boundary
                conditions for axes in which the grid is periodic and vanishing
                derivatives for all other axes. Alternatively, specific boundary
                conditions can be set for all axes individually. 
            bc2:
                The boundary condition for the second derivative,
                :math:`\nabla^2 u`.
        """
        super().__init__(noise=noise)
        
        self.nu = nu
        self.bc = bc
        self.bc2 = bc2
            
            
    def evolution_rate(self, state: ScalarField,  # type: ignore
                       t: float = 0) -> ScalarField:
        """ evaluate the right hand side of the PDE
        
        Args:
            state (:class:`~pde.fields.ScalarField`):
                The scalar field describing the concentration distribution
            t (float): The current time point
            
        Returns:
            :class:`~pde.fields.ScalarField`:
            Scalar field describing the evolution rate of the PDE 
        """
        assert isinstance(state, ScalarField)
        state_lap = state.laplace(bc=self.bc)
        result = -self.nu * state_lap.laplace(bc=self.bc2) - state_lap \
                 - 0.5 * state.gradient(bc=self.bc).to_scalar('squared_sum')
        result.label = 'evolution rate'
        return result  # type: ignore
    
    
    def _make_pde_rhs_numba(self, state: ScalarField  # type: ignore
                            ) -> Callable:
        """ create a compiled function evaluating the right hand side of the PDE
        
        Args:
            state (:class:`~pde.fields.ScalarField`):
                An example for the state defining the grid and data types
                
        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`numpy.ndarray` giving
            the evolution rate.  
        """
        shape = state.grid.shape
        dim = state.grid.dim
        arr_type = nb.typeof(np.empty(shape, dtype=np.double))
        signature = arr_type(arr_type, nb.double)
        
        nu_value = self.nu
        laplace = state.grid.get_operator('laplace', bc=self.bc)
        laplace2 = state.grid.get_operator('laplace', bc=self.bc2)
        gradient = state.grid.get_operator('gradient', bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """ compiled helper function evaluating right hand side """ 
            grad = gradient(state_data)
            result = -laplace(state_data)
            result += nu_value * laplace2(result)
            for i in range(dim):
                result -= 0.5 * grad[i]**2
            return result
            
        return pde_rhs  # type: ignore
    
    
    
