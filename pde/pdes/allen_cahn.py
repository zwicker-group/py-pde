"""
A Allen-Cahn equation

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable  # @UnusedImport

import numpy as np


from .base import PDEBase
from ..fields import ScalarField
from ..grids.boundaries.axes import BoundariesData
from ..tools.numba import nb, jit
from ..tools.docstrings import fill_in_docstring


        
class AllenCahnPDE(PDEBase):
    r""" A simple Allen-Cahn equation
    
    The mathematical definition is
    
    .. math::
        \partial_t c = \gamma \nabla^2 c - c^3 + c
        
    where :math:`c` is a scalar field and :math:`\gamma` sets the interfacial 
    width.
    """

    explicit_time_dependence = False


    @fill_in_docstring
    def __init__(self, interface_width: float = 1,
                 bc: BoundariesData = 'natural'):
        """ 
        Args:
            interface_width (float):
                The diffusivity of the described species
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
        """
        super().__init__()
        
        self.interface_width = interface_width
        self.bc = bc
            
            
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
        laplace = state.laplace(bc=self.bc, label='evolution rate')
        return self.interface_width * laplace - state**3 + state  # type: ignore
    
    
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
        arr_type = nb.typeof(np.empty(shape, dtype=np.double))
        signature = arr_type(arr_type, nb.double)
        
        interface_width = self.interface_width
        laplace = state.grid.get_operator('laplace', bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """ compiled helper function evaluating right hand side """ 
            return (interface_width * laplace(state_data) - state_data**3 +
                    state_data) 
            
        return pde_rhs  # type: ignore
    
    
    
