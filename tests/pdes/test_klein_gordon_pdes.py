"""
.. codeauthor:: Greg Partin <gpartin@gmail.com>
"""

import platform

import numpy as np
import pytest

from pde import PDE, KleinGordonPDE, ScalarField, UnitGrid
from pde.backends import backends
from pde.tools.misc import module_available


@pytest.mark.parametrize("dim", [1, 2])
def test_klein_gordon_consistency(dim, rng):
    """Test some methods of the Klein-Gordon model."""
    eq = KleinGordonPDE()
    assert isinstance(str(eq), str)
    assert isinstance(repr(eq), str)

    # prepare numpy implementation
    grid = UnitGrid([4] * dim)
    state = eq.get_initial_condition(ScalarField.random_uniform(grid, rng=rng))
    field = eq.evolution_rate(state)
    assert field.grid == grid

    # compare numba to numpy implementation
    if module_available("numba"):
        rhs = eq.make_pde_rhs(state, backend=backends["numba"])
        np.testing.assert_allclose(field.data, rhs(state.data, 0), rtol=1e-6)

    # compare torch to numpy implementation
    if module_available("torch") and platform.system() != "Windows":
        rhs = eq.make_pde_rhs(state, backend="torch")
        # use reduced tolerance to support potential float32 devices
        np.testing.assert_allclose(field.data, rhs(state.data, 0), rtol=1e-6)

    # compare to generic implementation
    assert isinstance(eq.expressions, dict)
    eq2 = PDE(eq.expressions)
    np.testing.assert_allclose(field.data, eq2.evolution_rate(state).data)


@pytest.mark.parametrize("dim", [1, 2])
def test_klein_gordon_reduces_to_wave(dim, rng):
    """Test that KleinGordonPDE with mass=0 reduces to the wave equation."""
    from pde import WavePDE

    speed = 2.0
    eq_kg = KleinGordonPDE(speed=speed, mass=0)
    eq_wave = WavePDE(speed=speed)

    grid = UnitGrid([4] * dim)
    u = ScalarField.random_uniform(grid, rng=rng)
    state_kg = eq_kg.get_initial_condition(u)
    state_wave = eq_wave.get_initial_condition(u.copy())

    rate_kg = eq_kg.evolution_rate(state_kg)
    rate_wave = eq_wave.evolution_rate(state_wave)
    np.testing.assert_allclose(rate_kg.data, rate_wave.data)


def test_klein_gordon_custom_params(rng):
    """Test KleinGordonPDE with custom speed and mass parameters."""
    eq = KleinGordonPDE(speed=2, mass=3)
    assert eq.speed == 2
    assert eq.mass == 3

    grid = UnitGrid([8])
    state = eq.get_initial_condition(ScalarField.random_uniform(grid, rng=rng))
    field = eq.evolution_rate(state)
    assert field.grid == grid

    # compare to generic implementation
    eq2 = PDE(eq.expressions)
    np.testing.assert_allclose(field.data, eq2.evolution_rate(state).data)
