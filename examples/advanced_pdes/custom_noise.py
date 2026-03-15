"""
Custom noise
============

This example solves a diffusion equation with a custom noise.
"""

import numpy as np

from pde import DiffusionPDE, ScalarField, UnitGrid


class DiffusionCustomNoisePDE(DiffusionPDE):
    """Diffusion PDE with custom noise implementation."""

    def make_noise_realization(self, state, backend):
        """Spatially-dependent noise scaled by the x-coordinate."""
        noise = float(self.noise)
        x_values = state.grid.cell_coords[..., 0]

        def noise_realization(state_data, t):
            return x_values * np.random.uniform(-noise, noise, size=state_data.shape)  # noqa: NPY002

        return noise_realization


eq = DiffusionCustomNoisePDE(diffusivity=0.1, noise=0.1)  # define the pde
state = ScalarField.random_uniform(UnitGrid([64, 64]))  # generate initial condition
result = eq.solve(state, t_range=10, dt=0.01)
result.plot()
