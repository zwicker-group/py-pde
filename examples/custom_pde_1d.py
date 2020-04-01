"""
Custom PDE in one dimension
===========================

This example implements a PDE that is only defined in one dimension.
Here, we chose the `Korteweg-de Vries equation
<https://en.wikipedia.org/wiki/Korteweg–de_Vries_equation>`_, given by

.. math::
    \partial_t \phi = 6 \phi \partial_x \phi - \partial_x^2 \phi
    
which we implement using a custom PDE class below.
"""

from math import pi

from pde import (CartesianGrid, ScalarField, PDEBase, MemoryStorage,
                 plot_kymograph)


class KortewegDeVriesPDE(PDEBase):
    """ Korteweg-de Vries equation """
    
    def evolution_rate(self, state, t=0):
        """ implement the python version of the evolution equation """
        assert state.grid.dim == 1  # ensure the state is one-dimensional
        grad = state.gradient('natural')[0]
        return 6 * state * grad - grad.laplace('natural')


# initialize the equation and the space
grid = CartesianGrid([[0, 2*pi]], [32], periodic=True)
state = ScalarField.from_expression(grid, "sin(x)")

# solve the equation and store the trajectory
storage = MemoryStorage()
eq = KortewegDeVriesPDE()             
eq.solve(state, t_range=3, tracker=storage.tracker(.1))

# plot the trajectory as a space-time plot
plot_kymograph(storage)
