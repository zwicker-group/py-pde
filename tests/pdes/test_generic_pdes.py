"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import ScalarField, UnitGrid, pdes
from pde.solvers import ExplicitSolver


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize(
    "pde_class",
    [
        pdes.AllenCahnPDE,
        pdes.CahnHilliardPDE,
        pdes.DiffusionPDE,
        pdes.KPZInterfacePDE,
        pdes.KuramotoSivashinskyPDE,
        pdes.SwiftHohenbergPDE,
    ],
)
def test_pde_consistency(pde_class, dim, rng):
    """Test some methods of generic PDE models."""
    eq = pde_class()
    assert isinstance(str(eq), str)
    assert isinstance(repr(eq), str)

    # compare numba to numpy implementation
    grid = UnitGrid([4] * dim)
    state = ScalarField.random_uniform(grid, rng=rng)
    field = eq.evolution_rate(state)
    assert field.grid == grid
    rhs = eq._make_pde_rhs_numba(state)
    res = rhs(state.data, 0)
    np.testing.assert_allclose(field.data, res)

    # compare to generic implementation
    assert isinstance(eq.expression, str)
    eq2 = pdes.PDE({"c": eq.expression})
    np.testing.assert_allclose(field.data, eq2.evolution_rate(state).data)


def test_pde_consistency_test(rng):
    """Test whether the consistency of a pde implementation is checked."""

    class TestPDE(pdes.PDEBase):
        def evolution_rate(self, field, t=0):
            return 2 * field

        def _make_pde_rhs_numba(self, state):
            def impl(state_data, t):
                return 3 * state_data

            return impl

    eq = TestPDE()
    state = ScalarField.random_uniform(UnitGrid([4]), rng=rng)
    with pytest.raises(AssertionError):
        eq.solve(state, t_range=5, tracker=None)


def test_pde_automatic_adaptive_solver():
    """Test whether adaptive solver is enabled as expected."""
    eq = pdes.DiffusionPDE()
    state = ScalarField(UnitGrid([2]))
    args = {"state": state, "t_range": 0.01, "backend": "numpy", "tracker": None}

    eq.solve(**args)
    assert eq.diagnostics["solver"]["dt_adaptive"]

    eq.solve(dt=0.01, **args)
    assert not eq.diagnostics["solver"]["dt_adaptive"]

    eq.solve(solver=ExplicitSolver, **args)
    assert not eq.diagnostics["solver"]["dt_adaptive"]

    eq.solve(solver="implicit", **args)
    assert not eq.diagnostics["solver"]["dt_adaptive"]
