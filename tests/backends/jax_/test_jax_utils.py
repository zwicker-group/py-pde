"""Tests for make_expression_function in the JAX backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from pde.tools.expressions import ScalarExpression


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_jax_expression_function_simple(backend):
    """Test make_expression_function with a simple expression."""
    expr = ScalarExpression("2 * x + 1", signature=["x"])
    func = backend.make_expression_function(expr)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = np.asarray(func(x))
    np.testing.assert_allclose(result, 2 * x + 1)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_jax_expression_function_single_arg(backend):
    """Test make_expression_function with single_arg=True."""
    expr = ScalarExpression("x + y", signature=["x", "y"])
    func = backend.make_expression_function(expr, single_arg=True)
    xy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = np.asarray(func(xy))
    expected = xy[0] + xy[1]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_jax_expression_function_heaviside(backend):
    """Test make_expression_function with Heaviside special function."""
    expr = ScalarExpression("Heaviside(x)", signature=["x"])
    func = backend.make_expression_function(expr)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    result = np.asarray(func(x))
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_jax_expression_function_hypot(backend):
    """Test make_expression_function with hypot special function."""
    expr = ScalarExpression("hypot(x, y)", signature=["x", "y"])
    func = backend.make_expression_function(expr)
    x = np.array([3.0], dtype=np.float32)
    y = np.array([4.0], dtype=np.float32)
    result = np.asarray(func(x, y))
    np.testing.assert_allclose(result, [5.0])


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_jax_expression_function_with_consts(backend):
    """Test make_expression_function with expression containing constants."""
    c = np.float32(3.0)
    expr = ScalarExpression("a * x", signature=["x"], consts={"a": c})
    func = backend.make_expression_function(expr)
    x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    result = np.asarray(func(x))
    np.testing.assert_allclose(result, c * x)
