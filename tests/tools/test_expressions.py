"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging

import numba as nb
import numpy as np
import pytest

from pde import FieldCollection, ScalarField, Tensor2Field, UnitGrid, VectorField
from pde.tools.expressions import (
    BCDataError,
    ScalarExpression,
    TensorExpression,
    evaluate,
    parse_expr_guarded,
    parse_number,
)


def test_parse_number():
    """test parse_number function"""
    assert parse_number(0) == pytest.approx(0)
    assert parse_number(1.235) == pytest.approx(1.235)
    assert parse_number("0") == pytest.approx(0)
    assert parse_number("1.235") == pytest.approx(1.235)
    assert parse_number("-1.4e-5") == pytest.approx(-1.4e-5)
    assert parse_number("sin(3.4)") == pytest.approx(np.sin(3.4))
    assert parse_number("sin(a)", {"a": 3}) == pytest.approx(np.sin(3))
    assert parse_number("a**b", {"a": 3.2, "b": 4.5}) == pytest.approx(3.2**4.5)
    assert parse_number("1 + 2 * I") == pytest.approx(1 + 2j)
    assert parse_number("a", {"a": 1 + 2j}) == pytest.approx(1 + 2j)
    with pytest.raises(TypeError):
        parse_number("foo")


def test_parse_expr_guarded():
    """test parse_expr_guarded function"""
    peg = parse_expr_guarded
    assert peg("1") == 1
    assert peg("1 + 1") == 2
    assert peg("a + 1").subs("a", 1) == 2
    with pytest.raises(TypeError):
        peg("field + 1")
    assert peg("field + 1", symbols=["field"]).subs("field", 1) == 2
    assert peg("field + 1", symbols=[None, ["field"]]).subs("field", 1) == 2
    assert peg("field + 1", symbols=["field", "a"]).subs("field", 1) == 2
    assert peg("field + 1", symbols={"field": 1, "a": 2}).subs("field", 1) == 2


def test_const(caplog):
    """test simple expressions"""
    # test scalar expressions with constants
    for expr in [None, 1, "1", "a - a"]:
        e = ScalarExpression() if expr is None else ScalarExpression(expr)
        val = 0 if expr is None or expr == "a - a" else float(expr)
        assert e.constant
        assert e.value == val
        assert e() == val
        assert e.get_compiled()() == val
        assert not e.depends_on("a")
        assert e.differentiate("a").value == 0
        assert e.shape == tuple()
        assert e.rank == 0
        assert bool(e) == (val != 0)
        assert e.is_zero == (val == 0)
        assert not e.complex

        g = e.derivatives
        assert g.constant
        assert isinstance(str(g), str)
        np.testing.assert_equal(g.value, [])

        for f in [
            ScalarExpression(e),
            ScalarExpression(e.expression),
            ScalarExpression(e.value),
        ]:
            assert e is not f
            assert e._sympy_expr == f._sympy_expr

    # test whether wrong constants are check for
    field = ScalarField(UnitGrid([3]))
    e = ScalarExpression("scalar_field", consts={"scalar_field": field})
    with caplog.at_level(logging.WARNING):
        assert e() == field
    assert "field" in caplog.text
    if not nb.config.DISABLE_JIT:  # @UndefinedVariable
        with pytest.raises(Exception):
            e.get_compiled()()


def test_single_arg(rng):
    """test simple expressions"""
    e = ScalarExpression("2 * a")
    assert not e.constant
    assert e.depends_on("a")
    assert e(4) == 8
    assert e.get_compiled()(4) == 8
    assert e.differentiate("a").value == 2
    assert e.differentiate("b").value == 0
    assert e.shape == tuple()
    assert e.rank == 0
    assert bool(e)
    assert not e.is_zero

    assert e == ScalarExpression(e.expression)
    with pytest.raises(TypeError):
        e.value

    arr = rng.random(5)
    np.testing.assert_allclose(e(arr), 2 * arr)
    np.testing.assert_allclose(e.get_compiled()(arr), 2 * arr)

    g = e.derivatives
    assert g.shape == (1,)
    assert g.constant
    assert g(3) == [2]
    assert g.get_compiled()(3) == [2]

    with pytest.raises(TypeError):
        ScalarExpression(np.exp)


def test_two_args(rng):
    """test simple expressions"""
    e = ScalarExpression("2 * a ** b")
    assert e.depends_on("b")
    assert not e.constant
    assert e(4, 2) == 32
    assert e.get_compiled()(4, 2) == 32
    assert e.differentiate("a")(4, 2) == 16
    assert e.differentiate("b")(4, 2) == pytest.approx(32 * np.log(4))
    assert e.differentiate("c").value == 0
    assert e.shape == tuple()
    assert e.rank == 0
    assert e == ScalarExpression(e.expression)

    for x in [rng.random(2), rng.random((2, 5))]:
        res = 2 * x[0] ** x[1]
        np.testing.assert_allclose(e(*x), res)
        np.testing.assert_allclose(e.get_compiled()(*x), res)
        if x.ndim == 1:
            func = e._get_function(single_arg=True)
            np.testing.assert_allclose(func(x), res)
            func = e.get_compiled(single_arg=True)
            np.testing.assert_allclose(func(x), res)

    g = e.derivatives
    assert g.shape == (2,)
    assert g.rank == 1
    assert not g.constant
    np.testing.assert_allclose(g(2, 3), [24, 16 * np.log(2)])
    np.testing.assert_allclose(g.get_compiled()(2, 3), [24, 16 * np.log(2)])


def test_derivatives():
    """test vector expressions"""
    e = ScalarExpression("a * b**2")
    assert e.depends_on("a") and e.depends_on("b")
    assert not e.constant
    assert e.rank == 0

    d1 = e.derivatives
    assert d1.shape == (2,)
    np.testing.assert_allclose(d1(2, 3), [9, 12])
    np.testing.assert_allclose(d1.get_compiled()(2, 3), [9, 12])

    d2 = d1.derivatives
    assert d2.shape == (2, 2)
    np.testing.assert_allclose(d2(2, 3), [[0, 6], [6, 4]])
    np.testing.assert_allclose(d2.get_compiled()(2, 3), [[0, 6], [6, 4]])

    d3 = d2.derivatives
    assert d3.shape == (2, 2, 2)

    d4 = d3.derivatives
    assert d4.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(d4(2, 3), np.zeros((2, 2, 2, 2)))
    np.testing.assert_allclose(d4.get_compiled()(2, 3), np.zeros((2, 2, 2, 2)))


def test_indexed():
    """test simple expressions"""
    e = ScalarExpression("2 * a[0] ** a[1]", allow_indexed=True)
    assert not e.constant
    assert e.depends_on("a")

    a = np.array([4, 2])
    assert e(a) == 32
    assert e.get_compiled()(a) == 32

    assert e.differentiate("a[0]")(a) == 16
    assert e.differentiate("a[1]")(a) == pytest.approx(32 * np.log(4))

    with pytest.raises(RuntimeError):
        e.differentiate("a")
    with pytest.raises(RuntimeError):
        e.derivatives


def test_synonyms(caplog):
    """test using synonyms in expression"""
    e = ScalarExpression("2 * arbitrary", [["a", "arbitrary"]])
    assert e.depends_on("a")
    assert not e.depends_on("arbitrary")


def test_tensor_expression():
    """test TensorExpression"""
    e = TensorExpression("[[0, 1], [2, 3]]")
    assert isinstance(str(e), str)
    assert e.shape == (2, 2)
    assert e.rank == 2
    assert e.constant
    np.testing.assert_allclose(e.get_compiled_array()(), [[0, 1], [2, 3]])
    np.testing.assert_allclose(e.get_compiled_array()(tuple()), [[0, 1], [2, 3]])
    assert e.differentiate("a") == TensorExpression("[[0, 0], [0, 0]]")
    np.testing.assert_allclose(e.value, np.arange(4).reshape(2, 2))

    e = TensorExpression("[a, 2*a]")
    assert isinstance(str(e), str)
    assert e.shape == (2,)
    assert e.rank == 1
    assert e.depends_on("a")
    assert not e.constant
    np.testing.assert_allclose(e.differentiate("a").value, np.array([1, 2]))
    with pytest.raises(TypeError):
        e.value
    assert e[0] == ScalarExpression("a")
    assert e[1] == ScalarExpression("2*a")
    assert e[0:1] == TensorExpression("[a]")
    np.testing.assert_allclose(e.get_compiled_array()(1.0), [1.0, 2.0])
    np.testing.assert_allclose(e.get_compiled_array()(2.0), [2.0, 4.0])

    e2 = TensorExpression(e)
    assert isinstance(str(e2), str)
    assert e == e2
    assert e is not e2


def test_expression_from_expression():
    """test creating expressions from expressions"""
    expr = ScalarExpression("sin(a)")
    assert expr == ScalarExpression(expr)
    assert expr != ScalarExpression(expr, ["a", "b"])
    with pytest.raises(RuntimeError):
        ScalarExpression(expr, "b")

    expr = ScalarExpression("sin(a)", ["a", "b"])
    assert expr == ScalarExpression(expr)
    assert expr != ScalarExpression(expr, "a")
    with pytest.raises(RuntimeError):
        ScalarExpression(expr, "b")


def test_expression_user_funcs():
    """test the usage of user_funcs"""
    expr = ScalarExpression("func()", user_funcs={"func": lambda: 1})
    assert expr() == 1
    assert expr.get_compiled()() == 1
    assert expr.value == 1

    expr = ScalarExpression("f(pi)", user_funcs={"f": np.sin})
    assert expr.constant
    assert expr() == pytest.approx(0)
    assert expr.get_compiled()() == pytest.approx(0)
    assert expr.value == pytest.approx(0)

    expr = TensorExpression("[0, f(pi)]", user_funcs={"f": np.sin})
    assert expr.constant
    np.testing.assert_allclose(expr(), np.array([0, 0]), atol=1e-14)
    np.testing.assert_allclose(expr.get_compiled()(), np.array([0, 0]), atol=1e-14)
    np.testing.assert_allclose(expr.value, np.array([0, 0]), atol=1e-14)


def test_complex_expression():
    """test expressions with complex numbers"""
    for s in ["sqrt(-1)", "I"]:
        expr = ScalarExpression(s)
        assert expr.complex
        assert expr.constant
        assert expr.value == pytest.approx(1j)

    expr = TensorExpression("[1, I]")
    assert expr.complex
    assert expr.constant
    assert expr.rank == 1
    assert expr.shape == (2,)
    np.testing.assert_allclose(expr.value, np.array([1, 1j]))

    expr = TensorExpression("[[1, -1], [I, 2]]")
    assert expr.complex
    assert expr.constant
    assert expr.rank == 2
    assert expr.shape == (2, 2)
    np.testing.assert_allclose(expr.value, np.array([[1, -1], [1j, 2]]))


@pytest.mark.parametrize(
    "expression, value",
    [("Heaviside(x)", 0.5), ("Heaviside(x, 0.75)", 0.75), ("heaviside(x, 0.75)", 0.75)],
)
def test_expression_heaviside(expression, value):
    """test special cases of expressions"""
    expr = ScalarExpression(expression)
    assert not expr.constant
    assert expr(-1) == 0
    assert expr(0) == value
    assert expr(1) == 1
    np.testing.assert_allclose(expr(np.array([-1, 0, 1])), np.array([0, value, 1]))

    f = expr.get_compiled()
    assert f(-1) == 0
    assert f(0) == value
    assert f(1) == 1
    np.testing.assert_allclose(f(np.array([-1, 0, 1])), np.array([0, value, 1]))


def test_expression_consts():
    """test the usage of consts"""
    expr = ScalarExpression("a", consts={"a": 1})
    assert expr.constant
    assert not expr.depends_on("a")
    assert expr() == 1
    assert expr.get_compiled()() == 1
    assert expr.value == 1

    expr = ScalarExpression("a + b", consts={"a": 1})
    assert not expr.constant
    assert not expr.depends_on("a") and expr.depends_on("b")
    assert expr(2) == 3
    assert expr.get_compiled()(2) == 3

    expr = ScalarExpression("a + b", consts={"a": np.array([1, 2])})
    assert not expr.constant
    np.testing.assert_allclose(expr(np.array([2, 3])), np.array([3, 5]))
    np.testing.assert_allclose(expr.get_compiled()(np.array([2, 3])), np.array([3, 5]))


def test_evaluate_func_scalar():
    """test the evaluate function with scalar fields"""
    grid = UnitGrid([2, 4])
    field = ScalarField.from_expression(grid, "x")

    res1 = evaluate("laplace(a)", {"a": field})
    res2 = field.laplace("natural")
    np.testing.assert_almost_equal(res1.data, res2.data)

    res = evaluate("a - x", {"a": field})
    np.testing.assert_almost_equal(res.data, 0)

    field_c = ScalarField.from_expression(grid, "1 + 1j")
    res = evaluate("c", {"c": field_c})
    assert res.dtype == "complex"
    np.testing.assert_almost_equal(res.data, field_c.data)

    res = evaluate("abs(a)**2", {"a": field_c})
    assert res.dtype == "float"
    np.testing.assert_allclose(res.data, 2)

    # edge case
    res = evaluate("sin", {"a": field_c}, consts={"sin": 3.14})
    assert res.dtype == "float"
    np.testing.assert_allclose(res.data, 3.14)


def test_evaluate_func_vector():
    """test the evaluate function with vector fields"""
    grid = UnitGrid([3])
    field = ScalarField.from_expression(grid, "x")
    vec = VectorField.from_expression(grid, ["x"])

    res = evaluate("inner(v, v)", {"v": vec})
    assert isinstance(res, ScalarField)
    np.testing.assert_almost_equal(res.data, grid.axes_coords[0] ** 2)

    res = evaluate("outer(v, v)", {"v": vec})
    assert isinstance(res, Tensor2Field)
    np.testing.assert_almost_equal(res.data, [[grid.axes_coords[0] ** 2]])

    assert isinstance(evaluate("gradient(a)", {"a": field}), VectorField)


def test_evaluate_func_invalid():
    """test the evaluate function with invalid data"""
    field = ScalarField.from_expression(UnitGrid([3]), "x")

    with pytest.raises(ValueError):
        evaluate("x", {})

    with pytest.raises(ValueError):
        evaluate("x", {"x": field})

    field2 = ScalarField.from_expression(UnitGrid([4]), "x")
    with pytest.raises(ValueError):
        evaluate("a + b", {"a": field, "b": field2})

    with pytest.raises(BCDataError):
        evaluate("laplace(a)", {"a": field}, bc=RuntimeError)

    with pytest.raises(RuntimeError):
        evaluate("a + b", {"a": field})


def test_evaluate_func_bcs_warning(caplog):
    """test whether a warning is thrown correctly"""
    field = ScalarField.from_expression(UnitGrid([3]), "x")

    with caplog.at_level(logging.WARNING):
        evaluate("laplace(u)", {"u": field}, bc_ops={"u:gradient": "value"})
    assert "gradient" in caplog.text

    with caplog.at_level(logging.WARNING):
        evaluate("laplace(u)", {"u": field}, bc_ops={"v:laplace": "value"})
    assert "Unused" in caplog.text


def test_evaluate_func_collection():
    """test the evaluate function with a field collection"""
    grid = UnitGrid([3])
    field = ScalarField.from_expression(grid, "x")
    vec = VectorField.from_expression(grid, ["x"])
    col = FieldCollection({"v": vec, "a": field})

    res = evaluate("inner(v, v)", col)
    assert isinstance(res, ScalarField)
    np.testing.assert_almost_equal(res.data, grid.axes_coords[0] ** 2)

    assert isinstance(evaluate("gradient(a)", col), VectorField)
    assert isinstance(evaluate("a * v", col), VectorField)

    with pytest.raises(RuntimeError):
        col.labels = ["a", "a"]
        evaluate("1", col)


def test_expression_repl(rng):
    """test expressions replacement"""
    e = ScalarExpression("2 * a", repl={"a": "b"})
    assert not e.constant
    assert e.depends_on("b")
    assert e(4) == 8
    assert e.differentiate("a").value == 0
    assert e.differentiate("b").value == 2

    e = ScalarExpression("2 * σ", repl={"σ": "sigma"})
    assert not e.constant
    assert e.depends_on("sigma")
    assert e(4) == 8
    assert e.differentiate("σ").value == 0
    assert e.differentiate("sigma").value == 2
