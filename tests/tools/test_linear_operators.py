"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import sympy

from pde.tools.linear_stability import make_linear_operator


def test_linop_basics():
    """Test basic operations."""
    from sympy.abc import a, x

    linop = make_linear_operator([x])

    assert linop(a).equals(0)
    assert linop(a + x).equals(linop(x))
    assert linop(a + 2 * x).equals(2 * linop(x))


def test_linop_series_1var():
    """Test series expansion of linear operators of one variable."""
    from sympy.abc import x

    f = sympy.Function("f")
    g = sympy.Function("g")
    x0 = sympy.Symbol("x0")
    linop = make_linear_operator([x])

    series = sympy.series(linop(x), x, x0, 2)
    res = sympy.simplify(series).removeO()
    expect = linop(x)
    assert res.equals(expect)

    series = sympy.series(linop(f(x)), x, x0, 2)
    res = sympy.simplify(series).removeO()
    expect = sympy.diff(f(x), x).subs({x: x0}) * linop(x)
    assert res.equals(expect)

    series = sympy.series(linop(f(x**2)), x, x0, 2)
    res = sympy.simplify(series).removeO()
    expect = sympy.diff(f(x**2), x).subs({x: x0}) * linop(x)
    assert res.equals(expect)

    series = sympy.series(linop(f(x) * linop(x)), x, x0, 2)
    res = sympy.simplify(series).removeO()
    expect = f(x0) * linop(linop(x))
    assert res.equals(expect)

    series = sympy.series(linop(f(x) * linop(g(x))), x, x0, 2)
    res = sympy.simplify(series).removeO()
    expect = f(x0) * sympy.diff(g(x), x).subs({x: x0}) * linop(linop(x))
    assert res.equals(expect)
