"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde import PDE, FieldCollection, ScalarField, SwiftHohenbergPDE, VectorField, grids


def iter_grids():
    """ generate some test grids """
    yield grids.UnitGrid([2, 2], periodic=[True, False])
    yield grids.CartesianGrid([[0, 1]], [2], periodic=[False])
    yield grids.CylindricalGrid(2, (0, 2), (2, 2), periodic_z=True)
    yield grids.SphericalGrid(2, 2)
    yield grids.PolarGrid(2, 2)


def test_pde_wrong_input():
    """ test some wrong input """
    with pytest.raises(RuntimeError):
        PDE({"t": 1})

    grid = grids.UnitGrid([4])
    eq = PDE({"u": 1})
    assert eq.expressions == {"u": "1.0"}
    with pytest.raises(ValueError):
        eq.evolution_rate(FieldCollection.scalar_random_uniform(2, grid))

    eq = PDE({"u": 1, "v": 2})
    assert eq.expressions == {"u": "1.0", "v": "2.0"}
    with pytest.raises(ValueError):
        eq.evolution_rate(ScalarField.random_uniform(grid))

    eq = PDE({"u": "a"})
    with pytest.raises(RuntimeError):
        eq.evolution_rate(ScalarField.random_uniform(grid))


def test_pde_scalar():
    """ test PDE with a single scalar field """
    eq = PDE({"u": "laplace(u) + exp(-t) + sin(t)"})
    assert eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8])
    field = ScalarField.random_normal(grid)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


def test_pde_vector():
    """ test PDE with a single vector field """
    eq = PDE({"u": "vector_laplace(u) + exp(-t)"})
    assert eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8, 8])
    field = VectorField.random_normal(grid)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


def test_pde_2scalar():
    """ test PDE with two scalar fields """
    eq = PDE({"u": "laplace(u) - u", "v": "- u * v"})
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8])
    field = FieldCollection.scalar_random_uniform(2, grid)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


@pytest.mark.slow
def test_pde_vector_scalar():
    """ test PDE with a vector and a scalar field """
    eq = PDE({"u": "vector_laplace(u) - u + gradient(v)", "v": "- divergence(u)"})
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8, 8])
    field = FieldCollection(
        [VectorField.random_uniform(grid), ScalarField.random_uniform(grid)]
    )

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


@pytest.mark.parametrize("grid", iter_grids())
def test_compare_swift_hohenberg(grid):
    """ compare custom class to swift-Hohenberg """
    rate, kc2, delta = np.random.uniform(0.5, 2, size=3)
    eq1 = SwiftHohenbergPDE(rate=rate, kc2=kc2, delta=delta)
    eq2 = PDE(
        {
            "u": f"({rate} - {kc2}**2) * u - 2 * {kc2} * laplace(u) "
            f"- laplace(laplace(u)) + {delta} * u**2 - u**3"
        }
    )
    assert eq1.explicit_time_dependence == eq2.explicit_time_dependence
    assert eq1.complex_valued == eq2.complex_valued

    field = ScalarField.random_uniform(grid)
    res1 = eq1.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res2 = eq2.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)

    res1.assert_field_compatible(res1)
    np.testing.assert_allclose(res1.data, res2.data)


def test_custom_operators():
    """ test using a custom operator """
    grid = grids.UnitGrid([32])
    field = ScalarField.random_normal(grid)
    eq = PDE({"u": "undefined(u)"})

    with pytest.raises(NameError):
        eq.evolution_rate(field)

    def make_op(state):
        return lambda state: state

    grids.UnitGrid.register_operator("undefined", make_op)

    eq._cache = {}  # reset cache
    res = eq.evolution_rate(field)
    np.testing.assert_allclose(field.data, res.data)

    del grids.UnitGrid._operators["undefined"]  # reset original state


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_noise(backend):
    """ test noise operator on PDE class """
    grid = grids.UnitGrid([64, 64])
    state = FieldCollection([ScalarField(grid), ScalarField(grid)])

    eq = PDE({"a": 0, "b": 0}, noise=0.5)
    res = eq.solve(state, t_range=1, backend=backend, dt=1, tracker=None)
    assert res.data.std() == pytest.approx(0.5, rel=0.1)

    eq = PDE({"a": 0, "b": 0}, noise=[0.01, 2.0])
    res = eq.solve(state, t_range=1, backend=backend, dt=1)
    assert res.data[0].std() == pytest.approx(0.01, rel=0.1)
    assert res.data[1].std() == pytest.approx(2.0, rel=0.1)

    with pytest.raises(ValueError):
        eq = PDE({"a": 0}, noise=[0.01, 2.0])
        eq.solve(ScalarField(grid), t_range=1, backend=backend, dt=1, tracker=None)


def test_pde_spatial_args():
    """ test ScalarFieldExpression without extra dependence """

    eq = PDE({"a": "x"})

    field = ScalarField(grids.UnitGrid([2]))
    rhs = eq.evolution_rate(field)
    assert rhs == field.copy(data=[0.5, 1.5])
    rhs = eq.make_pde_rhs(field, backend="numba")
    np.testing.assert_allclose(rhs(field.data, 0.0), np.array([0.5, 1.5]))

    eq = PDE({"a": "x + y"})
    with pytest.raises(RuntimeError):
        eq.evolution_rate(field)


def test_pde_user_funcs():
    """ test user supplied functions """
    # test a simple case
    eq = PDE({"u": "get_x(gradient(u))"}, user_funcs={"get_x": lambda arr: arr[0]})
    field = ScalarField.random_colored(grids.UnitGrid([32, 32]))
    rhs = eq.evolution_rate(field)
    np.testing.assert_allclose(rhs.data, field.gradient("natural").data[0])
    f = eq._make_pde_rhs_numba(field)
    np.testing.assert_allclose(f(field.data, 0), field.gradient("natural").data[0])


def test_pde_complex():
    """ test complex valued PDE """
    eq = PDE({"p": "I * laplace(p)"})
    assert not eq.explicit_time_dependence
    assert eq.complex_valued

    field = ScalarField.random_uniform(grids.UnitGrid([4]))
    assert not field.is_complex
    res1 = eq.solve(field, t_range=1, dt=0.1, backend="numpy", tracker=None)
    assert res1.is_complex
    res2 = eq.solve(field, t_range=1, dt=0.1, backend="numpy", tracker=None)
    assert res2.is_complex

    np.testing.assert_allclose(res1.data, res2.data)


def test_pde_product_operators():
    """ test inner and outer products """
    eq = PDE(
        {"p": "gradient(dot(p, p) + inner(p, p)) + tensor_divergence(outer(p, p))"}
    )
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    field = VectorField(grids.UnitGrid([4]), 1)
    res = eq.solve(field, t_range=1, dt=0.1, backend="numpy", tracker=None)
    np.testing.assert_allclose(res.data, field.data)


def test_pde_setting_noise():
    """ test setting the noise strength """
    for noise in [[0, 1], {"b": 1}, {"b": 1, "a": 0}, {"b": 1, "c": 1}]:
        eq = PDE({"a": "0", "b": "0"}, noise=noise)
        assert eq.is_sde
        assert eq.noise == [0, 1]

    for noise in [0, [0, 0]]:
        eq = PDE({"a": "0", "b": "0"}, noise=noise)
        assert not eq.is_sde

    with pytest.raises(ValueError):
        PDE({"a": 0}, noise=[1, 2])


def test_pde_consts():
    """ test using the consts argument in PDE """
    field = ScalarField(grids.UnitGrid([3]), 1)

    eq = PDE({"a": "b"}, consts={"b": 2})
    np.testing.assert_allclose(eq.evolution_rate(field).data, 2)

    eq = PDE({"a": "b**2"}, consts={"b": field})
    np.testing.assert_allclose(eq.evolution_rate(field).data, field.data)

    eq = PDE({"a": "laplace(b)"}, consts={"b": field})
    np.testing.assert_allclose(eq.evolution_rate(field).data, 0)

    eq = PDE({"a": "laplace(b)"}, consts={"b": 3})
    with pytest.raises(Exception):
        eq.evolution_rate(field)

    eq = PDE({"a": "laplace(b)"}, consts={"b": field.data})
    with pytest.raises(Exception):
        eq.evolution_rate(field)
