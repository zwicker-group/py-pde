'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from ..expressions import parse_number, ScalarExpression, TensorExpression



def test_parse_number():
    """ test parse_number function """
    assert parse_number(0) == pytest.approx(0)
    assert parse_number(1.235) == pytest.approx(1.235)
    assert parse_number("0") == pytest.approx(0)
    assert parse_number("1.235") == pytest.approx(1.235)
    assert parse_number("-1.4e-5") == pytest.approx(-1.4e-5)
    assert parse_number("sin(3.4)") == pytest.approx(np.sin(3.4))
    assert parse_number("sin(a)", {'a': 3}) == pytest.approx(np.sin(3))
    assert parse_number("a**b", {'a': 3.2, 'b': 4.5}) == pytest.approx(3.2**4.5)
    with pytest.raises(TypeError):
        parse_number('foo')



def test_const():
    """ test simple expressions """
    for expr in [None, 1, '1', 'a - a']:
        e = ScalarExpression() if expr is None else ScalarExpression(expr)
        val = 0 if expr is None or expr == 'a - a' else float(expr)
        assert e.constant
        assert e.value == val
        assert e() == val
        assert e.get_compiled()() == val
        assert not e.depends_on('a')
        assert e.differentiate('a').value == 0
        assert e.shape == tuple()
        assert e.rank == 0
        assert bool(e) == (val != 0)
        assert e.is_zero == (val == 0)
        
        g = e.derivatives
        assert g.constant
        np.testing.assert_equal(g.value, [])
            
        for f in [ScalarExpression(e), ScalarExpression(e.expression),
                  ScalarExpression(e.value)]:
            assert e is not f
            assert e._sympy_expr == f._sympy_expr



def test_single_arg():
    """ test simple expressions """
    e = ScalarExpression('2 * a')
    assert not e.constant
    assert e.depends_on('a')
    assert e(4) == 8
    assert e.get_compiled()(4) == 8
    assert e.differentiate('a').value == 2
    assert e.differentiate('b').value == 0
    assert e.shape == tuple()
    assert e.rank == 0
    assert bool(e)
    assert not e.is_zero
    
    assert e == ScalarExpression(e.expression)
    with pytest.raises(TypeError):
        e.value
    
    arr = np.random.random(5)
    np.testing.assert_allclose(e(arr), 2 * arr)
    np.testing.assert_allclose(e.get_compiled()(arr), 2 * arr)
    
    g = e.derivatives
    assert g.shape == (1,)
    assert g.constant
    assert g(3) == [2]
    assert g.get_compiled()(3) == [2]
    
    with pytest.raises(TypeError):
        ScalarExpression(np.exp)



def test_two_args():
    """ test simple expressions """
    e = ScalarExpression('2 * a ** b')
    assert e.depends_on('b')
    assert not e.constant
    assert e(4, 2) == 32
    assert e.get_compiled()(4, 2) == 32
    assert e.differentiate('a')(4, 2) == 16
    assert e.differentiate('b')(4, 2) == pytest.approx(32*np.log(4))
    assert e.differentiate('c').value == 0
    assert e.shape == tuple()
    assert e.rank == 0
    assert e == ScalarExpression(e.expression)

    for x in [np.random.random(2), np.random.random((2, 5))]:
        res = 2 * x[0]**x[1]
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
    """ test vector expressions """
    e = ScalarExpression('a * b**2')
    assert e.depends_on('a') and e.depends_on('b')
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
    np.testing.assert_allclose(d4.get_compiled()(2, 3),
                               np.zeros((2, 2, 2, 2)))

            
            
def test_indexed():
    """ test simple expressions """
    e = ScalarExpression('2 * a[0] ** a[1]', allow_indexed=True)
    assert not e.constant
    assert e.depends_on('a')
    
    a = np.array([4, 2])
    assert e(a) == 32
    assert e.get_compiled()(a) == 32
    
    assert e.differentiate('a[0]')(a) == 16
    assert e.differentiate('a[1]')(a) == pytest.approx(32*np.log(4))
    
    with pytest.raises(RuntimeError):
        e.differentiate('a')
    with pytest.raises(RuntimeError):
        e.derivatives
        
    
        
def test_synonyms():
    """ test using synonyms in expression """
    e = ScalarExpression('2 * all', [['a', 'all']])
    assert e.depends_on('a')
    assert not e.depends_on('all')
    


def test_tensor_expression():
    """ test TensorExpression """
    e = TensorExpression('[[0, 1], [2, 3]]')
    assert isinstance(str(e), str)
    assert e.shape == (2, 2)
    assert e.rank == 2
    assert e.constant
    assert e.differentiate('a') == TensorExpression('[[0, 0], [0, 0]]')
    np.testing.assert_allclose(e.value, np.arange(4).reshape(2, 2))
    
    e = TensorExpression('[a, 2*a]')
    assert isinstance(str(e), str)
    assert e.shape == (2,)
    assert e.rank == 1
    assert e.depends_on('a')
    assert not e.constant
    np.testing.assert_allclose(e.differentiate('a').value, np.array([1, 2]))
    with pytest.raises(TypeError):
        e.value
    
    e2 = TensorExpression(e)
    assert isinstance(str(e2), str)
    assert e == e2
    assert e is not e2

