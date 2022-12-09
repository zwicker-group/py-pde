"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDE, ScalarField, UnitGrid, WavePDE


@pytest.mark.parametrize("dim", [1, 2])
def test_wave_consistency(dim):
    """test some methods of the wave model"""
    eq = WavePDE()
    assert isinstance(str(eq), str)
    assert isinstance(repr(eq), str)

    # compare numba to numpy implementation
    grid = UnitGrid([4] * dim)
    state = eq.get_initial_condition(ScalarField.random_uniform(grid))
    field = eq.evolution_rate(state)
    assert field.grid == grid
    rhs = eq._make_pde_rhs_numba(state)
    np.testing.assert_allclose(field.data, rhs(state.data, 0))

    # compare to generic implementation
    assert isinstance(eq.expressions, dict)
    eq2 = PDE(eq.expressions)
    np.testing.assert_allclose(field.data, eq2.evolution_rate(state).data)
