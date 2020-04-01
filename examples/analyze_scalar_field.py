"""
Visualizing a scalar field
==========================

This example displays methods for visualizing scalar fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from pde import CylindricalGrid, ScalarField

# create a scalar field with some noise
grid = CylindricalGrid(7, [0, 4 * np.pi], 64)
data = ScalarField.from_expression(grid, 'sin(z) * exp(-r / 3)')
data += 0.05 * ScalarField.random_normal(grid)

# manipulate the field 
smoothed = data.smooth()  # Gaussian smoothing to get rid of the noise
projected = data.project('r')  # integrate along the radial direction
sliced = smoothed.slice({'z': 1})  # slice the smoothed data

# create four plots of the field and the modifications
fig, axes = plt.subplots(nrows=2, ncols=2)
data.plot(ax=axes[0, 0], title='Original field')
smoothed.plot(ax=axes[1, 0], title='Smoothed field')
projected.plot(ax=axes[0, 1], title='Projection on axial coordinate')
sliced.plot(ax=axes[1, 1], title='Slice of smoothed field at $z=1$')
plt.subplots_adjust(hspace=0.8)
