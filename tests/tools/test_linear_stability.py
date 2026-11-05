"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest
import sympy

from pde.tools.linear_stability import (
    Laplace,
    dispersion_relation_scalar,
    linearize_expression,
    linearize_expressions,
)


def test_linearize_expression():
    """Test linearization of a single expression."""
    u = sympy.Symbol("u")

    result = linearize_expression(u**2, {"u": 3}, check=False)
    assert result == 6 * u

    with pytest.raises(ValueError):
        linearize_expression(u**2, {"u": 3}, check=True)


def test_linearize_expression_with_operator():
    """Test linearization of an expression containing a differential operator."""
    expression = sympy.sympify(
        "laplace(u**2)", locals={"laplace": sympy.Function("laplace")}
    )
    u = sympy.Symbol("u")

    result = linearize_expression(expression, {"u": 3})

    assert result == 6 * Laplace(u)


def test_linearize_expressions_with_nested_functions_and_operators():
    """Test linearizing nonlinear expressions with nested functions and operators."""
    u, v = sympy.symbols("u v")
    laplace = sympy.Function("laplace")
    expressions = [
        laplace(u * v + sympy.sin(u)) + sympy.exp(u * v),
        laplace(v**2 + u) + sympy.sin(u * v),
    ]

    with pytest.raises(ValueError):
        linearize_expressions(expressions, {"u": 2, "v": 3}, check=True)

    results = linearize_expressions(expressions, {"u": 2, "v": 3}, check=False)
    expected = [
        (3 * u + 2 * v) * sympy.exp(6)
        + (sympy.cos(2) + 3) * Laplace(u)
        + 2 * Laplace(v),
        (3 * u + 2 * v) * sympy.cos(6) + Laplace(u) + 6 * Laplace(v),
    ]
    differences = [
        sympy.simplify(result - target)
        for result, target in zip(results, expected, strict=True)
    ]
    assert differences == [0, 0]


def test_linearize_expression_requires_single_state_variable():
    """Test that single-expression linearization requires one state variable."""
    u, v = sympy.symbols("u v")

    with pytest.raises(ValueError):
        linearize_expression(u + v, {"u": 1, "v": 2})


def test_linearize_expressions():
    """Test linearization of a system of expressions."""
    u, v = sympy.symbols("u v")

    with pytest.raises(ValueError):
        linearize_expressions([u**2 + u * v, v**2 + u], {"u": 2, "v": 3}, check=True)

    results = linearize_expressions(
        [u**2 + u * v, v**2 + u], {"u": 2, "v": 3}, check=False
    )
    assert results == [7 * u + 2 * v, u + 6 * v]


def test_linearize_expressions_requires_matching_state():
    """Test that the number of expressions matches the number of state variables."""
    u, v = sympy.symbols("u v")

    with pytest.raises(ValueError):
        linearize_expressions([u + v], {"u": 1, "v": 2})


def test_dispersion_relation_scalar_simple():
    """Test dispersion relation for a simple diffusive equation."""
    sympy.Symbol("u")
    laplace = sympy.Function("laplace")

    # ∂u/∂t = D∇²u  linearized around u=0 gives: growth = D*laplace(u)
    expression = sympy.sympify("D * laplace(u)", locals={"D": 1, "laplace": laplace})

    dispersion = dispersion_relation_scalar(expression, {"u": 0})

    # The growth rate should be -D*k² for diffusion
    # Evaluate at k² = 1
    result = dispersion(1)
    assert result == -1

    # Evaluate at k² = 4
    result = dispersion(4)
    assert result == -4


def test_dispersion_relation_scalar_with_parameter():
    """Test dispersion relation with a model parameter."""
    sympy.Symbol("u")
    D = sympy.Symbol("D", positive=True)
    laplace = sympy.Function("laplace")

    expression = sympy.sympify("D * laplace(u)", locals={"D": D, "laplace": laplace})

    dispersion = dispersion_relation_scalar(expression, {"u": 0})

    # At k² = 4, growth rate should be -4D
    result = dispersion(4)
    assert sympy.simplify(result + 4 * D) == 0

    # At k² = 1, growth rate should be -D
    result = dispersion(1)
    assert sympy.simplify(result + D) == 0


def test_dispersion_relation_scalar_diffusion_reaction():
    """Test dispersion relation for a reaction-diffusion system."""
    sympy.Symbol("u")
    laplace = sympy.Function("laplace")

    # ∂u/∂t = u(1 - u) + D∇²u
    # Linearized around u=1: ∂u/∂t = -u + D∇²u (where u is perturbation)
    # This gives growth rate: -1 - D*k²
    expression = sympy.sympify(
        "u * (1 - u) + D * laplace(u)",
        locals={"D": 1, "laplace": laplace},
    )

    dispersion = dispersion_relation_scalar(expression, {"u": 1})

    # Evaluate at k² = 1: should give -1 - 1 = -2
    result = dispersion(1)
    assert result == -2

    # Evaluate at k² = 0: should give -1 (no diffusion stabilization)
    result = dispersion(0)
    assert result == -1


def test_dispersion_relation_scalar_callable():
    """Test that the returned dispersion relation is callable."""
    sympy.Symbol("u")
    laplace = sympy.Function("laplace")

    expression = sympy.sympify("laplace(u)", locals={"laplace": laplace})

    dispersion = dispersion_relation_scalar(expression, {"u": 0})

    # Should be a Lambda (callable)
    assert isinstance(dispersion, sympy.Lambda)

    # Should be callable with k_sq
    result = dispersion(4)
    assert result == -4

    result = dispersion(9)
    assert result == -9


def test_dispersion_relation_scalar_returns_lambda():
    """Test that the returned object is a SymPy Lambda function."""
    sympy.Symbol("u")
    laplace = sympy.Function("laplace")

    expression = sympy.sympify("laplace(u)", locals={"laplace": laplace})

    result = dispersion_relation_scalar(expression, {"u": 0})

    assert isinstance(result, sympy.Lambda)
    assert len(result.args) == 2  # Lambda has signature and expression
