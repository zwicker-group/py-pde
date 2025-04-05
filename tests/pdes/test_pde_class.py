"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import os

import numba as nb
import numpy as np
import pytest
import sympy
from scipy import stats

from pde import PDE, MemoryStorage, SwiftHohenbergPDE, grids
from pde.fields import FieldCollection, ScalarField, VectorField
from pde.grids.boundaries.local import BCDataError


def iter_grids():
    """Generate some test grids."""
    yield grids.UnitGrid([2, 2], periodic=[True, False])
    yield grids.CartesianGrid([[0, 1]], [2], periodic=[False])
    yield grids.CylindricalSymGrid(2, (0, 2), (2, 2), periodic_z=True)
    yield grids.SphericalSymGrid(2, 2)
    yield grids.PolarSymGrid(2, 2)


def test_pde_critical_input(rng):
    """Test some wrong input and edge cases."""
    # test whether reserved symbols can be used as variables
    grid = grids.UnitGrid([4])
    eq = PDE({"E": 1})
    res = eq.solve(ScalarField(grid), t_range=2)
    t_final = eq.diagnostics["controller"]["t_final"]
    np.testing.assert_allclose(res.data, t_final)

    with pytest.raises(ValueError):
        PDE({"t": 1})

    eq = PDE({"u": 1})
    assert eq.expressions == {"u": "1.0"}
    with pytest.raises(ValueError):
        eq.evolution_rate(FieldCollection.scalar_random_uniform(2, grid))

    eq = PDE({"u": 1, "v": 2})
    assert eq.expressions == {"u": "1.0", "v": "2.0"}
    with pytest.raises(ValueError):
        eq.evolution_rate(ScalarField.random_uniform(grid, rng=rng))

    eq = PDE({"u": "a"})
    with pytest.raises(RuntimeError):
        eq.evolution_rate(ScalarField.random_uniform(grid, rng=rng))

    eq = PDE({"x": "x"})
    with pytest.raises(ValueError):
        eq.evolution_rate(ScalarField(grid))


def test_pde_scalar(rng):
    """Test PDE with a single scalar field."""
    eq = PDE({"u": "laplace(u) + exp(-t) + sin(t)"})
    assert eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8])
    field = ScalarField.random_normal(grid, rng=rng)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


def test_pde_vector(rng):
    """Test PDE with a single vector field."""
    eq = PDE({"u": "vector_laplace(u) + exp(-t)"})
    assert eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8, 8])
    field = VectorField.random_normal(grid, rng=rng).smooth(1)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


def test_pde_2scalar():
    """Test PDE with two scalar fields."""
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
def test_pde_vector_scalar(rng):
    """Test PDE with a vector and a scalar field."""
    eq = PDE({"u": "vector_laplace(u) - u + gradient(v)", "v": "- divergence(u)"})
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8, 8])
    field = FieldCollection(
        [
            VectorField.random_uniform(grid, rng=rng).smooth(1),
            ScalarField.random_uniform(grid, rng=rng).smooth(1),
        ]
    )

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


@pytest.mark.parametrize("grid", iter_grids())
def test_compare_swift_hohenberg(grid, rng):
    """Compare custom class to swift-Hohenberg."""
    rate, kc2, delta = rng.uniform(0.5, 2, size=3)
    eq1 = SwiftHohenbergPDE(rate=rate, kc2=kc2, delta=delta)
    eq2 = PDE(
        {
            "u": f"({rate} - {kc2}**2) * u - 2 * {kc2} * laplace(u) "
            f"- laplace(laplace(u)) + {delta} * u**2 - u**3"
        }
    )
    assert eq1.explicit_time_dependence == eq2.explicit_time_dependence
    assert eq1.complex_valued == eq2.complex_valued

    field = ScalarField.random_uniform(grid, rng=rng)
    res1 = eq1.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res2 = eq2.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)

    res1.assert_field_compatible(res1)
    np.testing.assert_allclose(res1.data, res2.data)


def test_custom_operators(rng):
    """Test using a custom operator."""
    grid = grids.UnitGrid([32])
    field = ScalarField.random_normal(grid, rng=rng)
    eq = PDE({"u": "undefined(u)"})

    with pytest.raises(ValueError):
        eq.evolution_rate(field)

    def make_op(state):
        def op(arr, out):
            out[:] = arr[1:-1]  # copy valid part of the array

        return op

    grids.UnitGrid.register_operator("undefined", make_op)

    eq._cache = {}  # reset cache
    res = eq.evolution_rate(field)
    np.testing.assert_allclose(field.data, res.data)

    del grids.UnitGrid._operators["undefined"]  # reset original state


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_noise(backend, rng):
    """Test noise operator on PDE class."""
    grid = grids.UnitGrid([128, 128])
    state = FieldCollection([ScalarField(grid), ScalarField(grid)])

    var_local = 0.5
    eq = PDE({"a": 0, "b": 0}, noise=var_local, rng=rng)
    res = eq.solve(state, t_range=1, backend=backend, dt=1, tracker=None)
    dist = stats.norm(scale=np.sqrt(var_local)).cdf
    assert stats.kstest(np.ravel(res.data), dist).pvalue > 0.001

    eq = PDE({"a": 0, "b": 0}, noise=[0.01, 2.0], rng=rng)
    res = eq.solve(state, t_range=1, backend=backend, dt=1, tracker=None)

    dist_a = stats.norm(scale=np.sqrt(0.01)).cdf
    dist_b = stats.norm(scale=np.sqrt(2)).cdf
    assert stats.kstest(np.ravel(res[0].data), dist_a).pvalue > 0.001
    assert stats.kstest(np.ravel(res[1].data), dist_b).pvalue > 0.001

    with pytest.raises(ValueError):
        eq = PDE({"a": 0}, noise=[0.01, 2.0])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_spatial_args(backend):
    """Test PDE with spatial dependence."""
    field = ScalarField(grids.UnitGrid([4]))

    eq = PDE({"a": "x"})
    rhs = eq.make_pde_rhs(field, backend=backend)
    np.testing.assert_allclose(rhs(field.data, 0.0), np.array([0.5, 1.5, 2.5, 3.5]))

    # test combination of spatial dependence and differential operators
    eq = PDE({"a": "dot(gradient(x), gradient(a))"})
    rhs = eq.make_pde_rhs(field, backend=backend)
    np.testing.assert_allclose(rhs(field.data, 0.0), np.array([0.0, 0.0, 0.0, 0.0]))

    # test invalid spatial dependence
    eq = PDE({"a": "x + y"})
    if backend == "numpy":
        with pytest.raises(RuntimeError):
            eq.evolution_rate(field)
    elif backend == "numba":
        with pytest.raises(RuntimeError):
            rhs = eq.make_pde_rhs(field, backend=backend)


def test_pde_user_funcs(rng):
    """Test user supplied functions."""
    # test a simple case
    eq = PDE({"u": "get_x(gradient(u))"}, user_funcs={"get_x": lambda arr: arr[0]})
    field = ScalarField.random_normal(grids.UnitGrid([32, 32]), rng=rng)
    rhs = eq.evolution_rate(field)
    np.testing.assert_allclose(
        rhs.data, field.gradient("auto_periodic_neumann").data[0]
    )
    f = eq._make_pde_rhs_numba(field)
    np.testing.assert_allclose(
        f(field.data, 0), field.gradient("auto_periodic_neumann").data[0]
    )


def test_pde_complex_serial(rng):
    """Test complex valued PDE."""
    eq = PDE({"p": "I * laplace(p)"})
    assert not eq.explicit_time_dependence
    assert eq.complex_valued

    field = ScalarField.random_uniform(grids.UnitGrid([4]), rng=rng)
    assert not field.is_complex
    res1 = eq.solve(field, t_range=1, dt=0.1, backend="numpy", tracker=None)
    assert res1.is_complex
    res2 = eq.solve(field, t_range=1, dt=0.1, backend="numba", tracker=None)
    assert res2.is_complex

    np.testing.assert_allclose(res1.data, res2.data)


def test_pde_product_operators():
    """Test inner and outer products."""
    eq = PDE(
        {"p": "gradient(dot(p, p) + inner(p, p)) + tensor_divergence(outer(p, p))"}
    )
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    field = VectorField(grids.UnitGrid([4]), 1)
    res = eq.solve(field, t_range=1, dt=0.1, backend="numpy", tracker=None)
    np.testing.assert_allclose(res.data, field.data)


def test_pde_setting_noise():
    """Test setting the noise strength."""
    for noise in [[0, 1], {"b": 1}, {"b": 1, "a": 0}, {"b": 1, "c": 1}]:
        eq = PDE({"a": "0", "b": "0"}, noise=noise)
        assert eq.is_sde
        np.testing.assert_allclose(eq.noise, [0, 1])

    for noise in [0, [0, 0]]:
        eq = PDE({"a": "0", "b": "0"}, noise=noise)
        assert not eq.is_sde

    with pytest.raises(ValueError):
        PDE({"a": 0}, noise=[1, 2])


def test_pde_consts():
    """Test using the consts argument in PDE."""
    field = ScalarField(grids.UnitGrid([3]), 1)

    eq = PDE({"a": "b"}, consts={"b": 2})
    np.testing.assert_allclose(eq.evolution_rate(field).data, 2)

    eq = PDE({"a": "b**2"}, consts={"b": field})
    np.testing.assert_allclose(eq.evolution_rate(field).data, field.data)

    eq = PDE({"a": "laplace(b)"}, consts={"b": field})
    np.testing.assert_allclose(eq.evolution_rate(field).data, 0)

    eq = PDE({"a": "laplace(b)"}, consts={"b": 3})
    with pytest.raises(
        AttributeError
        if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"
        else nb.TypingError
    ):
        eq.evolution_rate(field)

    eq = PDE({"a": "laplace(b)"}, consts={"b": field.data})
    with pytest.raises(TypeError):
        eq.evolution_rate(field)


@pytest.mark.parametrize("bc", ["auto_periodic_neumann", {"value": 1}])
def test_pde_bcs(bc, rng):
    """Test PDE with boundary conditions."""
    eq = PDE({"u": "laplace(u)"}, bc=bc)
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8])
    field = ScalarField.random_normal(grid, rng=rng)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


def test_pde_bcs_warning(caplog):
    """Test whether a warning is thrown correctly."""
    with caplog.at_level(logging.WARNING):
        eq = PDE({"u": "laplace(u)"}, bc_ops={"u:gradient": "value"})
        eq.evolution_rate(ScalarField(grids.UnitGrid([6])))
    assert "gradient" in caplog.text

    with caplog.at_level(logging.WARNING):
        eq = PDE({"u": "laplace(u)"}, bc_ops={"v:laplace": "value"})
        eq.evolution_rate(ScalarField(grids.UnitGrid([6])))
    assert "Unused" in caplog.text


@pytest.mark.parametrize("bc", ["asdf", [{"value": 1}] * 3])
def test_pde_bcs_error(bc, rng):
    """Test PDE with wrong boundary conditions."""
    eq = PDE({"u": "laplace(u)"}, bc=bc)
    grid = grids.UnitGrid([8, 8])
    field = ScalarField.random_normal(grid, rng=rng)

    for backend in ["numpy", "numba"]:
        with pytest.raises(BCDataError):
            eq.solve(field, t_range=1, dt=0.01, backend=backend, tracker=None)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_time_dependent_bcs(backend):
    """Test PDE with time-dependent BCs."""
    field = ScalarField(grids.UnitGrid([3]))

    eq = PDE({"c": "laplace(c)"}, bc={"value_expression": "Heaviside(t - 1.5)"})

    storage = MemoryStorage()
    eq.solve(field, t_range=10, dt=1e-2, backend=backend, tracker=storage.tracker(1))

    np.testing.assert_allclose(storage[1].data, 0)
    np.testing.assert_allclose(storage[-1].data, 1, rtol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_integral(backend, rng):
    """Test PDE with integral."""
    grid = grids.UnitGrid([16])
    field = ScalarField.random_uniform(grid, rng=rng)
    eq = PDE({"c": "-integral(c)"})

    # test rhs
    rhs = eq.make_pde_rhs(field, backend=backend)
    np.testing.assert_allclose(rhs(field.data, 0), -field.integral)

    # test evolution
    for method in ["scipy", "explicit"]:
        res = eq.solve(field, t_range=1000, solver=method, tracker=None)
        assert res.integral == pytest.approx(0, abs=1e-2)
        np.testing.assert_allclose(res.data, field.data - field.magnitude, atol=1e-3)


def test_anti_periodic_bcs():
    """Test a simulation with anti-periodic BCs."""
    grid = grids.CartesianGrid([[-10, 10]], 32, periodic=True)
    field = ScalarField.from_expression(grid, "0.01 * x**2")
    field -= field.average

    # test normal periodic BCs
    eq1 = PDE({"c": "laplace(c) + c - c**3"}, bc="periodic")
    res1 = eq1.solve(field, t_range=1e4, dt=1e-1)
    np.testing.assert_allclose(np.abs(res1.data), 1)
    assert res1.fluctuations == pytest.approx(0, abs=1e-5)

    # test normal anti-periodic BCs
    eq2 = PDE({"c": "laplace(c) + c - c**3"}, bc="anti-periodic")
    res2 = eq2.solve(field, t_range=1e3, dt=1e-3, adaptive=True)
    assert np.all(np.abs(res2.data) <= 1.0001)
    assert res2.fluctuations > 0.1


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_heaviside(backend):
    """Test PDE with a heaviside right hand side."""
    field = ScalarField(grids.CartesianGrid([[-1, 1]], 2), [-1, 1])
    eq = PDE({"c": "Heaviside(x)"})
    res = eq.solve(field, 0.999, dt=0.1, backend=backend, tracker=None)
    np.testing.assert_allclose(res.data, np.array([-1.0, 2]))


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_post_step_hook(backend):
    """Test simple post step hook function."""

    def post_step_hook(state_data, t):
        state_data[:5, :] = 1
        if t > 1:
            raise StopIteration

    eq = PDE({"c": "laplace(c)"}, post_step_hook=post_step_hook)
    state = ScalarField(grids.UnitGrid([10, 10]))
    result = eq.solve(state, dt=0.1, t_range=10, backend=backend, tracker=None)

    np.testing.assert_allclose(result.data[:5, :], 1)
    assert np.all(result.data[5:, :] > 0)
    assert np.all(result.data[5:, :] < 1)
    # Since the hook raises an exception the internal timer will not update and t_final
    # will be equal to the initial time
    assert eq.diagnostics["controller"]["t_final"] == 0


def test_jacobian_spectral():
    """Test jacobian_spectral method."""
    eq = PDE({"c": "laplace(c**3 - c - laplace(c))"})
    expected = "q**2*(-3*c**2 - q**2 + 1)"
    expr = eq._jacobian_spectral()[0]
    assert expr.expand() == sympy.parse_expr(expected).expand()
    expr = eq._jacobian_spectral(state_hom=1)
    np.testing.assert_equal(sympy.matrix2numpy(expr.subs("q", 1)), np.array([[-3]]))

    eq = PDE({"c": "divergence(c * gradient(c**3 - c - laplace(c)))"})
    expected = "2*c*q**2*(-2*c**2 - q**2 + 1)"
    expr = eq._jacobian_spectral()[0, 0]
    assert expr.expand() == sympy.parse_expr(expected).expand()
    expr = eq._jacobian_spectral(state_hom=1)
    np.testing.assert_equal(sympy.matrix2numpy(expr.subs("q", 1)), np.array([[-4]]))

    eq = PDE({"a": "laplace(a) + (b - 1)**2", "b": "laplace(a + b**2)"})
    jac = eq._jacobian_spectral(state_hom=[0, 1])
    assert jac[0, 0].expand() == sympy.parse_expr("-q**2").expand()
    assert jac[0, 1].expand() == sympy.parse_expr("0").expand()
    assert jac[1, 0].expand() == sympy.parse_expr("-q**2").expand()
    assert jac[1, 1].expand() == sympy.parse_expr("-2*q**2").expand()


def test_jacobian_spectral_bad_input():
    """Test jacobian_spectral method with bad input."""
    with pytest.raises(ValueError):
        PDE({"a": "a"})._jacobian_spectral(wave_vector="t")
    with pytest.raises(ValueError):
        PDE({"a": "a"})._jacobian_spectral(wave_vector="a")
    with pytest.raises(TypeError):
        PDE({"a": "a"})._jacobian_spectral(state_hom="1")
    with pytest.raises(TypeError):
        PDE({"a": "a"})._jacobian_spectral(state_hom=[np.arange(3)])
    with pytest.raises(ValueError):
        PDE({"a": "a"})._jacobian_spectral(state_hom=np.arange(2))
    with pytest.raises(RuntimeError):
        PDE({"a": "a"})._jacobian_spectral(state_hom=0.1)
    with pytest.raises(TypeError):
        PDE({"a": "inner(a, a)"})._jacobian_spectral(state_hom=0)
    with pytest.raises(TypeError):
        PDE({"a": "b"}, consts={"b": 1})._jacobian_spectral(state_hom=0)


def test_dispersion_relation():
    """Test dispersion_relation method."""
    eq = PDE({"c": "laplace(c**3 - c - laplace(c))"})
    qs, evs = eq._dispersion_relation(state_hom=0, qs=[0, 0.5, 1])
    np.testing.assert_allclose(qs, np.array([0, 0.5, 1]))
    np.testing.assert_allclose(evs[:, 0], np.array([0, 3 / 16, 0]))

    eq = PDE({"a": "laplace(a) + (b - 1)**2", "b": "laplace(a + b**2)"})
    qs, evs = eq._dispersion_relation(state_hom=[0, 1], qs=[0, 0.5, 1])
    np.testing.assert_allclose(qs, np.array([0, 0.5, 1]))
    np.testing.assert_allclose(evs[0], np.array([0, 0]))
    np.testing.assert_allclose(np.sort(evs[1]), np.array([-0.5, -0.25]))
    np.testing.assert_allclose(np.sort(evs[2]), np.array([-2, -1]))
