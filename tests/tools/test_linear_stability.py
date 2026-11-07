"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
import sympy as sp

from pde.tools.linear_stability import (
    Laplace,
    dispersion_matrix,
    dispersion_relation_scalar,
    linearize_expression,
    linearize_expressions,
)


def test_linearize_expression():
    """Test linearization of a single expression."""
    u = sp.Symbol("u")

    result = linearize_expression(u**2, {"u": 3}, check=False)
    assert result == 6 * u

    with pytest.raises(ValueError):
        linearize_expression(u**2, {"u": 3}, check=True)


def test_linearize_expression_with_operator():
    """Test linearization of an expression containing a differential operator."""
    result = linearize_expression("laplace(u**2)", {"u": 3})
    assert result == 6 * Laplace(sp.Symbol("u"))


def test_linearize_expressions_with_nested_functions_and_operators():
    """Test linearizing nonlinear expressions with nested functions and operators."""
    u, v = sp.symbols("u v")
    laplace = sp.Function("laplace")
    expressions = [
        laplace(u * v + sp.sin(u)) + sp.exp(u * v),
        laplace(v**2 + u) + sp.sin(u * v),
    ]

    with pytest.raises(ValueError):
        linearize_expressions(expressions, {"u": 2, "v": 3}, check=True)

    results = linearize_expressions(expressions, {"u": 2, "v": 3}, check=False)
    expected = [
        (3 * u + 2 * v) * sp.exp(6) + (sp.cos(2) + 3) * Laplace(u) + 2 * Laplace(v),
        (3 * u + 2 * v) * sp.cos(6) + Laplace(u) + 6 * Laplace(v),
    ]
    differences = [
        sp.simplify(result - target)
        for result, target in zip(results, expected, strict=True)
    ]
    assert differences == [0, 0]


def test_linearize_expression_requires_single_state_variable():
    """Test that single-expression linearization requires one state variable."""
    u, v = sp.symbols("u v")

    with pytest.raises(ValueError):
        linearize_expression(u + v, {"u": 1, "v": 2})


def test_linearize_expressions():
    """Test linearization of a system of expressions."""
    u, v = sp.symbols("u v")

    with pytest.raises(ValueError):
        linearize_expressions([u**2 + u * v, v**2 + u], {"u": 2, "v": 3}, check=True)

    results = linearize_expressions(
        [u**2 + u * v, v**2 + u], {"u": 2, "v": 3}, check=False
    )
    assert results == [7 * u + 2 * v, u + 6 * v]


def test_linearize_expressions_requires_matching_state():
    """Test that the number of expressions matches the number of state variables."""
    u, v = sp.symbols("u v")

    with pytest.raises(ValueError):
        linearize_expressions([u + v], {"u": 1, "v": 2})


@pytest.mark.parametrize("as_string", [True, False])
@pytest.mark.parametrize("ret_type", ["sympy", "numpy"])
def test_dispersion_matrix_input_and_output_types(as_string, ret_type):
    """Test string and SymPy inputs with both dispersion matrix output types."""
    if as_string:
        expressions = [
            "-u + laplace(u) + 2*v",
            "3*u - 2*v + 4*laplace(v)",
        ]
    else:
        u, v = sp.symbols("u v")
        expressions = [
            -u + Laplace(u) + 2 * v,
            3 * u - 2 * v + 4 * Laplace(v),
        ]

    dispersion = dispersion_matrix(expressions, {"u": 0, "v": 0}, ret_type=ret_type)

    if ret_type == "sympy":
        k_sq = sp.Symbol("k_sq")
        expected = sp.Matrix([[-1 - k_sq, 2], [3, -2 - 4 * k_sq]])
        assert isinstance(dispersion, sp.Lambda)
        assert dispersion(k_sq) == expected
    else:
        expected = np.array(
            [[[-1, 2], [3, -2]], [[-2, 2], [3, -6]]],
        )
        np.testing.assert_array_equal(dispersion(np.array([0, 1])), expected)


def test_dispersion_relation_scalar_simple():
    """Test dispersion relation for a simple diffusive equation."""
    laplace = sp.Function("laplace")

    # ∂u/∂t = D∇²u  linearized around u=0 gives: growth = D*laplace(u)
    expression = sp.sympify("D * laplace(u)", locals={"D": 1, "laplace": laplace})

    dispersion = dispersion_relation_scalar(expression, {"u": 0})
    assert isinstance(dispersion, sp.Lambda)
    assert len(dispersion.args) == 2  # Lambda has signature and expression

    # The growth rate should be -D*k² for diffusion
    # Evaluate at k² = 1
    result = dispersion(1)
    assert result == -1

    # Evaluate at k² = 4
    result = dispersion(4)
    assert result == -4


def test_dispersion_relation_scalar_with_parameter():
    """Test dispersion relation with a model parameter."""
    D = sp.Symbol("D", positive=True)
    laplace = sp.Function("laplace")
    expression = sp.sympify("D * laplace(u)", locals={"D": D, "laplace": laplace})

    dispersion = dispersion_relation_scalar(expression, {"u": 0})
    assert isinstance(dispersion, sp.Lambda)
    assert len(dispersion.args) == 2  # Lambda has signature and expression

    # At k² = 4, growth rate should be -4D
    result = dispersion(4)
    assert sp.simplify(result + 4 * D) == 0

    # At k² = 1, growth rate should be -D
    result = dispersion(1)
    assert sp.simplify(result + D) == 0


def test_dispersion_relation_scalar_with_complex_coefficients():
    """Test dispersion relation with complex reaction and diffusion coefficients."""
    u = sp.Symbol("u")
    laplace = sp.Function("laplace")
    expression = (1 + sp.I) * u + (2 - 3 * sp.I) * laplace(u)

    dispersion = dispersion_relation_scalar(expression, {"u": 0})

    k_sq = sp.Symbol("k_sq")
    expected = 1 + sp.I - (2 - 3 * sp.I) * k_sq
    assert sp.simplify(dispersion(k_sq) - expected) == 0
    assert dispersion(2) == -3 + 7 * sp.I


@pytest.mark.parametrize("ret_type", ["sympy", "numpy"])
def test_dispersion_relation_scalar_return_type(ret_type):
    """Test the requested output type of a scalar dispersion relation."""
    u = sp.Symbol("u")
    expression = (1 + sp.I) * u + (2 - 3 * sp.I) * Laplace(u)

    dispersion = dispersion_relation_scalar(expression, {"u": 0}, ret_type=ret_type)

    if ret_type == "sympy":
        k_sq = sp.Symbol("k_sq")
        expected = 1 + sp.I - (2 - 3 * sp.I) * k_sq
        assert isinstance(dispersion, sp.Lambda)
        assert sp.simplify(dispersion(k_sq) - expected) == 0
    else:
        result = dispersion(np.array([0, 2]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1 + 1j, -3 + 7j]))


def test_dispersion_relation_scalar_invalid_return_type():
    """Test that scalar dispersion relations reject unknown output types."""
    with pytest.raises(ValueError, match="Unknown `ret_type`"):
        dispersion_relation_scalar("laplace(u)", {"u": 0}, ret_type="unknown")


def test_dispersion_relation_scalar_diffusion_reaction():
    """Test dispersion relation for a reaction-diffusion system."""
    laplace = sp.Function("laplace")

    # ∂u/∂t = u(1 - u) + D∇²u
    # Linearized around u=1: ∂u/∂t = -u + D∇²u (where u is perturbation)
    # This gives growth rate: -1 - D*k²
    expression = sp.sympify(
        "u * (1 - u) + D * laplace(u)",
        locals={"D": 1, "laplace": laplace},
    )

    dispersion = dispersion_relation_scalar(expression, {"u": 1})
    assert isinstance(dispersion, sp.Lambda)
    assert len(dispersion.args) == 2  # Lambda has signature and expression

    # Evaluate at k² = 1: should give -1 - 1 = -2
    result = dispersion(1)
    assert result == -2

    # Evaluate at k² = 0: should give -1 (no diffusion stabilization)
    result = dispersion(0)
    assert result == -1


def test_dispersion_relation_scalar_callable():
    """Test that the returned dispersion relation is callable."""
    laplace = sp.Function("laplace")
    expression = sp.sympify("laplace(u)", locals={"laplace": laplace})

    dispersion = dispersion_relation_scalar(expression, {"u": 0})

    # Should be a Lambda (callable)
    assert isinstance(dispersion, sp.Lambda)
    assert isinstance(dispersion, sp.Lambda)
    assert len(dispersion.args) == 2  # Lambda has signature and expression

    # Should be callable with k_sq
    result = dispersion(4)
    assert result == -4

    result = dispersion(9)
    assert result == -9


def test_dispersion_relation_complex():
    """Test a complex case of a dispersion relation"""
    result = dispersion_relation_scalar(
        "u - sqrt(1 - u**2) + laplace(u**3 - u - sqrt(u)*laplace(u))",
        {"u": 1 / sp.sqrt(2)},
        check=True,
    )
    q2 = sp.Symbol("q2")
    assert result(q2).equals(2 - q2 / 2 - q2**2 * sp.parse_expr("2**-(1/4)"))


# TODO: test unsupported functions (undefined, integral, gradient, dot, constant fields)
