#!/usr/bin/env python3

from math import pi

from pde.grids import CartesianGrid
from pde.fields import ScalarField
from pde.storage import MemoryStorage
from pde.pdes import PDEBase
from pde.visualization import plot_kymograph


class KortewegDeVriesPDE(PDEBase):
    """ Korteweg-de Vries equation
    
    See https://en.wikipedia.org/wiki/Korteweg–de_Vries_equation
    """
    
    def evolution_rate(self, state, t=0):
        """ implement the python version of the evolution equation """
        assert state.grid.dim == 1  # ensure the state is one-dimensional
        grad = state.gradient('natural')[0]
        return 6 * state * grad - grad.laplace('natural')


# initialize the equation and the space
grid = CartesianGrid([[0, 2*pi]], [32], periodic=True)
state = ScalarField.from_expression(grid, "sin(x)")
eq = KortewegDeVriesPDE()             

# solve the equation and store the trajectory
storage = MemoryStorage()
eq.solve(state, t_range=3, tracker=['progress', storage.tracker(.1)])

# plot the trajectory as a space-time plot
plot_kymograph(storage, show=True)
