"""
Custom noise
============

This example solves a diffusion equation with a custom noise.
"""

from pde import DiffusionPDE, ScalarField, UnitGrid


class DiffusionCustomNoisePDE(DiffusionPDE):
    """Diffusion PDE with custom noise implementation."""

    use_noise_variance = True

    def make_noise_variance(self, state, *, backend, ret_diff=False):
        """Make function that calculates noise variance."""
        noise = float(self.noise)
        x_values = state.grid.cell_coords[..., 0]

        def noise_variance(state_data, t):
            return noise * x_values**2

        return noise_variance


eq = DiffusionCustomNoisePDE(diffusivity=0.1, noise=0.1)  # define the pde
state = ScalarField.random_uniform(UnitGrid([64, 64]))  # generate initial condition
result = eq.solve(state, t_range=10, dt=0.01)
result.plot()
