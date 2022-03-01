"""
1D problem - Using `PDE` class
==============================

This example implements a PDE that is only defined in one dimension.
Here, we chose the `Korteweg-de Vries equation
<https://en.wikipedia.org/wiki/Korteweg–de_Vries_equation>`_, given by

.. math::
    \partial_t \phi = 6 \phi \partial_x \phi - \partial_x^3 \phi
    
which we implement using the :class:`~pde.pdes.pde.PDE`.
"""

from math import pi

from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph

# initialize the equation and the space
eq = PDE({"φ": "6 * φ * d_dx(φ) - laplace(d_dx(φ))"})
grid = CartesianGrid([[0, 2 * pi]], [32], periodic=True)
state = ScalarField.from_expression(grid, "sin(x)")

# solve the equation and store the trajectory
storage = MemoryStorage()
eq.solve(state, t_range=3, tracker=storage.tracker(0.1))

# plot the trajectory as a space-time plot
plot_kymograph(storage)
