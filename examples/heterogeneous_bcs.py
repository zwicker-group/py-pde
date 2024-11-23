r"""
Heterogeneous boundary conditions
=================================

This example implements a `spatially coupled SIR model 
<https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology>`_ with 
the following dynamics for the density of susceptible, infected, and recovered
individuals:

.. math::

    \partial_t s &= D \nabla^2 s - \beta is \\
    \partial_t i &= D \nabla^2 i + \beta is - \gamma i \\
    \partial_t r &= D \nabla^2 r + \gamma i

Here, :math:`D` is the diffusivity, :math:`\beta` the infection rate, and
:math:`\gamma` the recovery rate.
"""

import numpy as np

from pde import CartesianGrid, DiffusionPDE, ScalarField

# define grid and an initial state
grid = CartesianGrid([[-5, 5], [-5, 5]], 32)
field = ScalarField(grid)


# define the boundary conditions, which here are calculated from a function
def bc_value(adjacent_value, dx, x, y, t):
    """Return boundary value."""
    return np.sign(x)


# define and solve a simple diffusion equation
eq = DiffusionPDE(bc={"*": {"derivative": 0}, "y+": {"value_expression": bc_value}})
res = eq.solve(field, t_range=10, dt=0.01, backend="numpy")
res.plot()
