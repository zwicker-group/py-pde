r"""
Simple custom PDE class
=======================

This example implements a custom scalar PDE. To display some of the features of
the package, we here implemented the `Kuramoto–Sivashinsky equation
<https://en.wikipedia.org/wiki/Kuramoto–Sivashinsky_equation>`_, which was
considered for flame front dynamics:

.. math::
    \partial_t u = -\frac12 |\nabla u|^2 - \nabla^2 u - \nabla^4 u
"""

from pde import UnitGrid, ScalarField, PDEBase


class KuramotoSivashinskyPDE(PDEBase):
    """ Implementation of the normalized Kuramoto–Sivashinsky equation """
    
    def evolution_rate(self, state, t=0):
        """ implement the python version of the evolution equation """
        state_lap = state.laplace(bc='natural')
        state_lap2 = state_lap.laplace(bc='natural')
        state_grad = state.gradient(bc='natural')
        return -state_grad.to_scalar('squared_sum') / 2 - state_lap - state_lap2 


grid = UnitGrid([16, 16])  # generate grid
state = ScalarField.random_uniform(grid)  # generate initial condition

eq = KuramotoSivashinskyPDE()  # define the pde
result = eq.solve(state, t_range=10, dt=0.01)
result.plot(show=True)
