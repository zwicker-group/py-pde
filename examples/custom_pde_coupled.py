r"""
Custom Class for coupled PDEs
=============================

This example shows how to solve a set of coupled PDEs, the
spatially coupled `FitzHugh–Nagumo model 
<https://en.wikipedia.org/wiki/FitzHugh–Nagumo_model>`_, which is a simple model
for the excitable dynamics of coupled Neurons:

.. math::

    \partial_t u &= \nabla^2 u + u (u - \alpha) (1 - u) + w \\
    \partial_t w &= \epsilon u
    
Here, :math:`\alpha` denotes the external stimulus and :math:`\epsilon` defines
the recovery time scale. We implement this as a custom PDE class below.
"""

from pde import UnitGrid, FieldCollection, PDEBase


class FitzhughNagumoPDE(PDEBase):
    """ FitzHugh–Nagumo model with diffusive coupling """
    
    def __init__(self, stimulus=.1, ε=1, bc='natural'):
        self.bc = bc
        self.stimulus = stimulus
        self.ε = ε
                
    def evolution_rate(self, state, t=0):
        u, w = state  # membrane potential and recovery variable
        
        u_t = u.laplace(bc=self.bc) + u * (u - self.stimulus) * (1 - u) + w
        w_t = self.ε * u
        
        return FieldCollection([u_t, w_t])


grid = UnitGrid([16, 16])
state = FieldCollection.scalar_random_uniform(2, grid)

eq = FitzhughNagumoPDE()
result = eq.solve(state, t_range=10, dt=0.01)
result.plot(show=True)
