"""
Plot a polar grid
=================

This example shows how to initialize a polar grid with a hole inside and angular
symmetry, so that fields only depend on the radial coordinate.
"""

from pde import PolarSymGrid

grid = PolarSymGrid((2, 5), 8)
grid.plot(title=f"Area={grid.volume:.5g}")
