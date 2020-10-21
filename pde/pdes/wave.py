"""
A simple diffusion equation

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Callable, Dict

import numpy as np

from ..fields import FieldCollection, ScalarField
from ..grids.boundaries.axes import BoundariesData
from ..tools.docstrings import fill_in_docstring
from ..tools.numba import jit, nb
from .base import PDEBase, expr_prod


class WavePDE(PDEBase):
    r""" A simple wave equation
    
    The mathematical definition,
    
    .. math::
        \partial_t^2 u = c^2 \nabla^2 u
        
    is implemented as two first-order equations:
    
    .. math::
        \partial_t u &= v \\
        \partial_t v &= c^2 \nabla^2 u
        
        
    where :math:`u` is the density field that and :math:`c` sets the wave speed.
    """

    explicit_time_dependence = False

    @fill_in_docstring
    def __init__(self, speed: float = 1, bc: BoundariesData = "natural"):
        """
        Args:
            speed (float):
                The speed :math:`c` of the wave
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
        """
        super().__init__()

        self.speed = speed
        self.bc = bc

    def get_initial_condition(self, u: ScalarField, v: ScalarField = None):
        """create a suitable initial condition

        Args:
            u (:class:`~pde.fields.ScalarField`):
                The initial density on the grid
            v (:class:`~pde.fields.ScalarField`, optional):
                The initial rate of change. This is assumed to be zero if the
                value is omitted.

        Returns:
            :class:`~pde.fields.FieldCollection`:
                The combined fields u and v, suitable for the simulation
        """
        if v is None:
            v = u.copy(data=0)
        return FieldCollection([u, v])

    @property
    def expressions(self) -> Dict[str, str]:
        """ dict: the expressions of the right hand side of this PDE """
        return {"u": "v", "v": expr_prod(self.speed ** 2, "laplace(u)")}

    def evolution_rate(  # type: ignore
        self,
        state: FieldCollection,
        t: float = 0,
    ) -> FieldCollection:
        """evaluate the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields :math:`u` and :math:`v` distribution
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.FieldCollection`:
            Scalar field describing the evolution rate of the PDE
        """
        assert isinstance(state, FieldCollection)
        u, v = state
        u_t = v.copy()
        v_t = self.speed ** 2 * u.laplace(self.bc)  # type: ignore
        return FieldCollection([u_t, v_t])

    def _make_pde_rhs_numba(self, state: FieldCollection) -> Callable:  # type: ignore
        """create a compiled function evaluating the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called
            with an instance of :class:`numpy.ndarray` of the state data and
            the time to obtained an instance of :class:`numpy.ndarray` giving
            the evolution rate.
        """
        shape = state.grid.shape
        arr_type = nb.typeof(np.empty((2,) + shape, dtype=state.data.dtype))
        signature = arr_type(arr_type, nb.double)

        speed2 = self.speed ** 2
        laplace = state.grid.get_operator("laplace", bc=self.bc)

        @jit(signature)
        def pde_rhs(state_data: np.ndarray, t: float):
            """ compiled helper function evaluating right hand side """
            rate = np.empty_like(state_data)
            rate[0] = state_data[1]
            laplace(state_data[0], out=rate[1])
            rate[1] *= speed2
            return rate

        return pde_rhs  # type: ignore
