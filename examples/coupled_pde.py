#!/usr/bin/env python3

from pde.grids import UnitGrid
from pde.fields import FieldCollection
from pde.pdes.base import PDEBase


class FitzhughNagumoPDE(PDEBase):
    """ FitzHugh–Nagumo model with diffusive coupling
    
    See https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model
    """
    
    def __init__(self, stimulus=.1, ε=1, bc='natural'):
        self.bc = bc
        self.stimulus = stimulus
        self.ε = ε
                
    def evolution_rate(self, state, t=0):
        u, w = state  # membrane potential and recovery variable
        
        u_t = u.laplace(bc=self.bc) + u * (u - self.stimulus) * (1 - u) + w
        w_t = self.ε * u
        
        return FieldCollection([u_t, w_t])


eq = FitzhughNagumoPDE()
grid = UnitGrid([16, 16])
state = FieldCollection.scalar_random_uniform(2, grid)

result = eq.solve(state, t_range=10, dt=0.01)
result.plot(show=True)
