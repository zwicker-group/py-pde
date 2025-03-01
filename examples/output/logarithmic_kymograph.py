r"""
Logarithmic kymograph
=====================

This example demonstrates a space-time plot with a logarithmic time axis, which is useful
to analyze coarsening processes. Here, we use :func:`utilitiez.densityplot` for plotting.
"""

import matplotlib.pyplot as plt
from utilitiez import densityplot

import pde

# define grid, initial field, and the PDE
grid = pde.UnitGrid([128])
field = pde.ScalarField.random_uniform(grid, -0.1, 0.1)
eq = pde.CahnHilliardPDE(interface_width=2)

# run the simulation and store data in logarithmically spaced time intervals
storage = pde.MemoryStorage()
res = eq.solve(
    field, t_range=1e5, adaptive=True, tracker=[storage.tracker("geometric(10, 1.1)")]
)

# create the density plot, which detects the logarithmically scaled time
densityplot(storage.data, storage.times, grid.axes_coords[0])
plt.xlabel("Time")
plt.ylabel("Space")
