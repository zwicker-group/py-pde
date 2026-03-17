"""Tests for numba-specific spherical grid operator features.

These tests cover functionality that is unique to the numba backend for
spherical grids. Generic operator tests that are shared across backends live in
``tests/backends/generic/operators/``.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde import ScalarField, SphericalSymGrid, Tensor2Field, VectorField


def test_conservative_sph_tensor():
    """Test tensor double divergence conservation (numba-specific)."""
    grid = SphericalSymGrid((0, 2), 50)
    expr = "1 / cosh((r - 1) * 10)"

    expressions = [[expr, 0, 0], [0, expr, 0], [0, 0, expr]]
    tf = Tensor2Field.from_expression(grid, expressions)
    res = tf.apply_operator(
        "tensor_double_divergence", bc="derivative", conservative=True, backend="numba"
    )
    assert res.integral == pytest.approx(0, abs=1e-3)
