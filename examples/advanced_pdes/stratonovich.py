"""
SDE with Stratonovich interpretation
====================================

This example solves a stochastic diffusion equation with Stratonovich interpretation
"""

from pde import PDE, ScalarField, UnitGrid


class AllenCahnNoisePDE(PDE):
    """Allen–Cahn PDE with custom noise implementation."""

    use_noise_variance = True

    def make_noise_variance(self, state, *, backend, ret_diff=False):
        """Make function that calculates noise variance."""
        noise = float(self.noise)

        if ret_diff:

            def noise_variance(state_data, t):
                return noise * state_data**2, 2 * noise * state_data

        else:

            def noise_variance(state_data, t):
                return noise * state_data**2

        return noise_variance


eq = AllenCahnNoisePDE(
    rhs={"c": "laplace(c) + c - c**3"}, noise=1.0, noise_interpretation="stratonovich"
)
state = ScalarField.random_uniform(UnitGrid([64, 64]), -1, 1)
result = eq.solve(state, t_range=10, dt=1e-3, solver="milstein")
result.plot()
