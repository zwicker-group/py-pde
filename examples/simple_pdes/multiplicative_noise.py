"""
Multiplicative noise
====================

This example solves an Allen-Cahn equation where the variance of the noise depends on
the field itself. Since the variance is given as an expression, its derivative is
determined automatically, so the equation can be interpreted in the anti-Itô sense
and integrated using the Milstein solver.
"""

from pde import PDE, ScalarField, UnitGrid

eq = PDE(
    {"c": "laplace(c) + c - c**3"},
    noise={"c": "0.1 * c**2"},
    noise_interpretation="anti-ito",
)
state = ScalarField.random_uniform(UnitGrid([64, 64]), -1, 1)
result = eq.solve(state, t_range=10, dt=1e-3, solver="milstein")
result.plot()
