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
        x_values_flat = state.grid.cell_coords[..., 0].ravel()
        flat_size = state.data.size
        data_shape = state.data.shape

        def noise_realization(state_data, t):
            return (  # noqa: NPY002
                x_values_flat * np.random.uniform(-noise, noise, flat_size)
            ).reshape(data_shape)

        return backend.compile_function(noise_realization)


eq = DiffusionCustomNoisePDE(diffusivity=0.1, noise=0.1)  # define the pde
state = ScalarField.random_uniform(UnitGrid([64, 64]))  # generate initial condition
result = eq.solve(state, t_range=10, dt=0.01)
result.plot()
