"""
Custom noise
============

This example solves a diffusion equation with a custom noise.
"""

import numpy as np

from pde import DiffusionPDE, ScalarField, UnitGrid
from pde.tools.numba import jit


class DiffusionCustomNoisePDE(DiffusionPDE):
    """Diffusion PDE with custom noise implementations."""

    def noise_realization(self, state, t):
        """Numpy implementation of spatially-dependent noise."""
        noise_field = ScalarField.random_uniform(state.grid, -self.noise, self.noise)
        return state.grid.cell_coords[..., 0] * noise_field

    def _make_noise_realization_numba(self, state):
        """Numba implementation of spatially-dependent noise."""
        noise = float(self.noise)
        x_values = state.grid.cell_coords[..., 0]

        @jit
        def noise_realization(state_data, t):
            return x_values * np.random.uniform(-noise, noise, size=state_data.shape)

        return noise_realization


eq = DiffusionCustomNoisePDE(diffusivity=0.1, noise=0.1)  # define the pde
state = ScalarField.random_uniform(UnitGrid([64, 64]))  # generate initial condition
result = eq.solve(state, t_range=10, dt=0.01)
result.plot()
