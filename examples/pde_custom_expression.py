r"""
Kuramoto-Sivashinsky - Using `PDE` class
========================================

This example implements a scalar PDE using the :class:`~pde.pdes.pde.PDE`. We here
consider the `Kuramoto–Sivashinsky equation
<https://en.wikipedia.org/wiki/Kuramoto–Sivashinsky_equation>`_, which for instance 
describes the dynamics of flame fronts:

.. math::
    \partial_t u = -\frac12 |\nabla u|^2 - \nabla^2 u - \nabla^4 u
"""

from pde import PDE, ScalarField, UnitGrid

grid = UnitGrid([32, 32])  # generate grid
state = ScalarField.random_uniform(grid)  # generate initial condition

eq = PDE({"u": "-gradient_squared(u) / 2 - laplace(u + laplace(u))"})  # define the pde
result = eq.solve(state, t_range=10, dt=0.01)
result.plot()
