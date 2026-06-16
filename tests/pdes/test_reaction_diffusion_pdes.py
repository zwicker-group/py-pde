"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging

import numpy as np
import pytest

from pde import (
    PDE,
    DiffusionPDE,
    FieldCollection,
    ReactionDiffusionPDE,
    ScalarField,
    UnitGrid,
)


def test_RD_diffusion_equivalence(rng):
    """Compare reaction-diffusion system to pure diffusion."""
    eq1 = ReactionDiffusionPDE(["c"], 2.5, {})
    eq2 = DiffusionPDE(diffusivity=2.5)
    assert not eq1.explicit_time_dependence

    grid = UnitGrid([4, 4])
    state = ScalarField.random_uniform(grid, rng=rng)

    rhs1 = eq1.evolution_rate(FieldCollection([state]))
    rhs2 = eq2.evolution_rate(state)
    np.testing.assert_allclose(rhs1[0].data, rhs2.data)


def test_RD_explicit_comparison(rng):
    """Explicitly compare RD system to PDE implementation."""
    # define the PDE
    a, b = 1, 3
    d0, d1 = 1, 0.1
    eq1 = ReactionDiffusionPDE(
        variables=["u", "v"],
        diffusivity=[d0, d1],
        sources=[f"{a} - ({b} + 1) * u + u**2 * v", f"{b} * u - u**2 * v"],
    )
    eq2 = PDE(
        {
            "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
            "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
        }
    )

    # initialize state
    grid = UnitGrid([8])
    state = FieldCollection.scalar_random_uniform(2, grid, 1, 2, rng=rng)

    rhs1 = eq1.evolution_rate(state)
    rhs2 = eq2.evolution_rate(state)
    np.testing.assert_allclose(rhs1.data, rhs2.data)


def test_RD_input(caplog):
    """Test various inputs to RD system."""
    with pytest.raises(ValueError):
        ReactionDiffusionPDE(["a", "a"], 1, {})
    with pytest.raises(ValueError):
        ReactionDiffusionPDE(["a"], [1, 2], {})
    with pytest.raises(ValueError):
        ReactionDiffusionPDE(["a"], 1, ["0", "1"])
    caplog.set_level(logging.WARNING)
    ReactionDiffusionPDE(["a"], 1, {"a": 0, "misspelled": 1})
    assert "misspelled" in caplog.text
