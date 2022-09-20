"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import PDE, MemoryStorage, SwiftHohenbergPDE, grids
from pde.fields import FieldCollection, ScalarField, VectorField
from pde.grids.boundaries.local import BCDataError
from pde.tools import mpi


def iter_grids():
    """generate some test grids"""
    yield grids.UnitGrid([2, 2], periodic=[True, False])
    yield grids.CartesianGrid([[0, 1]], [2], periodic=[False])
    yield grids.CylindricalSymGrid(2, (0, 2), (2, 2), periodic_z=True)
    yield grids.SphericalSymGrid(2, 2)
    yield grids.PolarSymGrid(2, 2)


def test_pde_critical_input():
    """test some wrong input and edge cases"""
    # test whether reserved symbols can be used as variables
    grid = grids.UnitGrid([4])
    eq = PDE({"E": 1})
    res = eq.solve(ScalarField(grid), t_range=2)
    np.testing.assert_allclose(res.data, 2)

    with pytest.raises(ValueError):
        PDE({"t": 1})

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

    eq = PDE({"x": "x"})
    with pytest.raises(ValueError):
        eq.evolution_rate(ScalarField(grid))


def test_pde_scalar():
    """test PDE with a single scalar field"""
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
    """test PDE with a single vector field"""
    eq = PDE({"u": "vector_laplace(u) + exp(-t)"})
    assert eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8, 8])
    field = VectorField.random_normal(grid).smooth(1)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


def test_pde_2scalar():
    """test PDE with two scalar fields"""
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
    """test PDE with a vector and a scalar field"""
    eq = PDE({"u": "vector_laplace(u) - u + gradient(v)", "v": "- divergence(u)"})
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8, 8])
    field = FieldCollection(
        [
            VectorField.random_uniform(grid).smooth(1),
            ScalarField.random_uniform(grid).smooth(1),
        ]
    )

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


@pytest.mark.parametrize("grid", iter_grids())
def test_compare_swift_hohenberg(grid):
    """compare custom class to swift-Hohenberg"""
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
    """test using a custom operator"""
    grid = grids.UnitGrid([32])
    field = ScalarField.random_normal(grid)
    eq = PDE({"u": "undefined(u)"})

    with pytest.raises(NameError):
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
def test_pde_noise(backend):
    """test noise operator on PDE class"""
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


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_spatial_args(backend):
    """test PDE with spatial dependence"""
    field = ScalarField(grids.UnitGrid([2]))

    eq = PDE({"a": "x"})
    rhs = eq.make_pde_rhs(field, backend=backend)
    np.testing.assert_allclose(rhs(field.data, 0.0), np.array([0.5, 1.5]))

    # test combination of spatial dependence and differential oeprators
    eq = PDE({"a": "dot(gradient(x), gradient(a))"})
    rhs = eq.make_pde_rhs(field, backend=backend)
    np.testing.assert_allclose(rhs(field.data, 0.0), np.array([0.0, 0.0]))

    # test invalid spatial dependence
    eq = PDE({"a": "x + y"})
    with pytest.raises(RuntimeError):
        rhs = eq.make_pde_rhs(field, backend=backend)
        rhs(field.data, 0.0)


def test_pde_user_funcs():
    """test user supplied functions"""
    # test a simple case
    eq = PDE({"u": "get_x(gradient(u))"}, user_funcs={"get_x": lambda arr: arr[0]})
    field = ScalarField.random_colored(grids.UnitGrid([32, 32]))
    rhs = eq.evolution_rate(field)
    np.testing.assert_allclose(
        rhs.data, field.gradient("auto_periodic_neumann").data[0]
    )
    f = eq._make_pde_rhs_numba(field)
    np.testing.assert_allclose(
        f(field.data, 0), field.gradient("auto_periodic_neumann").data[0]
    )


def test_pde_complex_serial():
    """test complex valued PDE"""
    eq = PDE({"p": "I * laplace(p)"})
    assert not eq.explicit_time_dependence
    assert eq.complex_valued

    field = ScalarField.random_uniform(grids.UnitGrid([4]))
    assert not field.is_complex
    res1 = eq.solve(field, t_range=1, dt=0.1, backend="numpy", tracker=None)
    assert res1.is_complex
    res2 = eq.solve(field, t_range=1, dt=0.1, backend="numba", tracker=None)
    assert res2.is_complex

    np.testing.assert_allclose(res1.data, res2.data)


@pytest.mark.multiprocessing
def test_pde_complex_mpi():
    """test complex valued PDE"""
    eq = PDE({"p": "I * laplace(p)"})
    assert not eq.explicit_time_dependence
    assert eq.complex_valued

    field = ScalarField.random_uniform(grids.UnitGrid([4]))
    assert not field.is_complex

    args = {
        "state": field,
        "t_range": 1.01,
        "dt": 0.1,
        "tracker": None,
        "ret_info": True,
    }
    res1, info1 = eq.solve(backend="numpy", method="explicit_mpi", **args)
    res2, info2 = eq.solve(backend="numba", method="explicit_mpi", **args)

    if mpi.is_main:
        # check results in the main process
        expect, _ = eq.solve(backend="numpy", method="explicit", **args)

        assert res1.is_complex
        np.testing.assert_allclose(res1.data, expect.data)
        assert info1["solver"]["steps"] == 11

        assert res2.is_complex
        np.testing.assert_allclose(res2.data, expect.data)
        assert info2["solver"]["steps"] == 11


def test_pde_product_operators():
    """test inner and outer products"""
    eq = PDE(
        {"p": "gradient(dot(p, p) + inner(p, p)) + tensor_divergence(outer(p, p))"}
    )
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    field = VectorField(grids.UnitGrid([4]), 1)
    res = eq.solve(field, t_range=1, dt=0.1, backend="numpy", tracker=None)
    np.testing.assert_allclose(res.data, field.data)


def test_pde_setting_noise():
    """test setting the noise strength"""
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
    """test using the consts argument in PDE"""
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


@pytest.mark.parametrize("bc", ["auto_periodic_neumann", {"value": 1}])
def test_pde_bcs(bc):
    """test PDE with boundary conditions"""
    eq = PDE({"u": "laplace(u)"}, bc=bc)
    assert not eq.explicit_time_dependence
    assert not eq.complex_valued
    grid = grids.UnitGrid([8])
    field = ScalarField.random_normal(grid)

    res_a = eq.solve(field, t_range=1, dt=0.01, backend="numpy", tracker=None)
    res_b = eq.solve(field, t_range=1, dt=0.01, backend="numba", tracker=None)

    res_a.assert_field_compatible(res_b)
    np.testing.assert_allclose(res_a.data, res_b.data)


@pytest.mark.parametrize("bc", ["asdf", [{"value": 1}] * 3])
def test_pde_bcs_error(bc):
    """test PDE with wrong boundary conditions"""
    eq = PDE({"u": "laplace(u)"}, bc=bc)
    grid = grids.UnitGrid([8, 8])
    field = ScalarField.random_normal(grid)

    for backend in ["numpy", "numba"]:
        with pytest.raises(BCDataError):
            eq.solve(field, t_range=1, dt=0.01, backend=backend, tracker=None)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_time_dependent_bcs(backend):
    """test PDE with time-dependent BCs"""
    field = ScalarField(grids.UnitGrid([3]))

    eq = PDE({"c": "laplace(c)"}, bc={"value_expression": "Heaviside(t - 1.5)"})

    storage = MemoryStorage()
    eq.solve(field, t_range=10, dt=1e-2, backend=backend, tracker=storage.tracker(1))

    np.testing.assert_allclose(storage[1].data, 0)
    np.testing.assert_allclose(storage[-1].data, 1, rtol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_pde_integral(backend):
    """test PDE with integral"""
    grid = grids.UnitGrid([16])
    field = ScalarField.random_uniform(grid)
    eq = PDE({"c": "-integral(c)"})

    # test rhs
    rhs = eq.make_pde_rhs(field, backend=backend)
    np.testing.assert_allclose(rhs(field.data, 0), -field.integral)

    # test evolution
    for method in ["scipy", "explicit"]:
        res = eq.solve(field, t_range=1000, method=method, tracker=None)
        assert res.integral == pytest.approx(0, abs=1e-2)
        np.testing.assert_allclose(res.data, field.data - field.magnitude, atol=1e-3)


def test_anti_periodic_bcs():
    """test a simulation with anti-periodic BCs"""
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
