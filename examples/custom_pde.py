#!/usr/bin/env python3

from pde.grids import UnitGrid
from pde.fields import ScalarField
from pde.pdes.base import PDEBase


class KuramotoSivashinskyPDE(PDEBase):
    """ Kuramoto–Sivashinsky equation
    
    See https://en.wikipedia.org/wiki/Kuramoto–Sivashinsky_equation
    """
    
    def evolution_rate(self, state, t=0):
        """ implement the python version of the evolution equation """
        state_lap = state.laplace(bc='natural')
        state_lap2 = state_lap.laplace(bc='natural')
        state_grad = state.gradient(bc='natural')
        return -state_grad.to_scalar('squared_sum') / 2 - state_lap - state_lap2 


eq = KuramotoSivashinskyPDE()             # define the pde
grid = UnitGrid([16, 16])                 # generate grid
state = ScalarField.random_uniform(grid)  # generate initial condition

result = eq.solve(state, t_range=10, dt=0.01)
result.plot(show=True)
