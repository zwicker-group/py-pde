"""
A Cahn-Hilliard equation

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable

import numpy as np


from .base import PDEBase
from ..fields import ScalarField
from ..grids.boundaries.axes import BoundariesData
from ..tools.numba import nb, jit
from ..tools.docstrings import fill_in_docstring


        
class CahnHilliardPDE(PDEBase):
    r""" A simple Cahn-Hilliard equation
    
    The mathematical definition is

    .. math::
        \partial_t c = \nabla^2 \left(c^3 - c - \gamma \nabla^2 c\right)
        
    where :math:`c` is a scalar field and :math:`\gamma` sets the interfacial 
    width.
    """

    explicit_time_dependence = False
    

    @fill_in_docstring
    def __init__(self, interface_width: float = 1,
                 bc_c: BoundariesData = 'natural',
                 bc_mu: BoundariesData = 'natural'):
        """ 
        Args:
            interface_width (float):
                The diffusivity of the described species
            bc_c:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            bc_mu:
                The boundary conditions applied to the chemical potential 
                associated with the scalar field :math:`c`. Supports the same
                options as `bc_c`.
        """
        super().__init__()
        
        self.interface_width = interface_width
        self.bc_c = bc_c
        self.bc_mu = bc_mu
            
            
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
        c_laplace = state.laplace(bc=self.bc_c, label='evolution rate')
        result = state**3 - state - self.interface_width * c_laplace
        return result.laplace(bc=self.bc_mu)  # type: ignore
    
    
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
        laplace_c = state.grid.get_operator('laplace', bc=self.bc_c)
        laplace_mu = state.grid.get_operator('laplace', bc=self.bc_mu)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """ compiled helper function evaluating right hand side """ 
            mu = (state_data**3 - state_data -
                  interface_width * laplace_c(state_data))
            return laplace_mu(mu) 
            
        return pde_rhs  # type: ignore
    
    
    
