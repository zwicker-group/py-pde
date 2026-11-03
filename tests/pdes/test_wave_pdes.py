"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform

import numpy as np
import pytest

from pde import PDE, ScalarField, UnitGrid, WavePDE, get_backend
from pde.tools.misc import module_available


@pytest.mark.parametrize("dim", [1, 2])
def test_wave_consistency(dim, rng):
    """Test some methods of the wave model."""
    eq = WavePDE()
    assert isinstance(str(eq), str)
    assert isinstance(repr(eq), str)

    # prepare numpy implementation
    grid = UnitGrid([4] * dim)
    state = eq.get_initial_condition(ScalarField.random_uniform(grid, rng=rng))
    field = eq.evolution_rate(state)
    assert field.grid == grid

    # compare numba to numpy implementation
    if module_available("numba"):
        rhs = eq.make_pde_rhs(state, backend="numba")
        res = rhs(state.data, 0)
        np.testing.assert_allclose(field.data, res, rtol=1e-6)

    # compare torch to numpy implementation
    if module_available("torch") and platform.system() != "Windows":
        rhs = eq.make_pde_rhs(state, backend="torch")
        res = get_backend("torch")._apply_function(rhs, state.data, 0)
        # use reduced tolerance to support potential float32 devices
        np.testing.assert_allclose(field.data, res, rtol=1e-6)

    # compare to generic implementation
    assert isinstance(eq.expressions, dict)
    eq2 = PDE(eq.expressions)
    np.testing.assert_allclose(field.data, eq2.evolution_rate(state).data)
