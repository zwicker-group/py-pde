"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import stats

from pde import CartesianGrid, DiffusionPDE, MemoryStorage, ScalarField, UnitGrid


def test_diffusion_single(rng):
    """Test some methods of the simple diffusion model."""
    eq = DiffusionPDE()
    assert isinstance(str(eq), str)
    assert isinstance(repr(eq), str)
    assert not eq.explicit_time_dependence

    grid = UnitGrid([4, 4])
    state = ScalarField.random_uniform(grid, rng=rng)

    field = eq.evolution_rate(state)
    assert isinstance(field, ScalarField)
    assert field.grid == grid


def test_simple_diffusion_value(rng):
    """Test a simple diffusion equation with constant boundaries."""
    grid = CartesianGrid([[0, 1]], [16])
    c = ScalarField.random_uniform(grid, 0, 1, rng=rng)
    eq = DiffusionPDE(bc={"x-": {"value": 0}, "x+": {"value": 1}})
    sol, info = eq.solve(c, t_range=1, dt=0.001, tracker=None, ret_info=True)
    assert isinstance(info, dict)
    np.testing.assert_allclose(sol.data, grid.axes_coords[0], rtol=5e-3)


def test_simple_diffusion_flux_right(rng):
    """Test a simple diffusion equation with flux boundary on the right."""
    grid = CartesianGrid([[0, 1]], [16])
    c = ScalarField.random_uniform(grid, 0, 1, rng=rng)
    eq = DiffusionPDE(bc={"x-": {"value": 0}, "x+": {"derivative": 3}})
    sol = eq.solve(c, t_range=5, dt=0.001, tracker=None)
    np.testing.assert_allclose(sol.data, 3 * grid.axes_coords[0], rtol=5e-3)


def test_simple_diffusion_flux_left(rng):
    """Test a simple diffusion equation with flux boundary on the left."""
    grid = CartesianGrid([[0, 1]], [16])
    c = ScalarField.random_uniform(grid, 0, 1, rng=rng)
    eq = DiffusionPDE(bc={"x-": {"derivative": 2}, "x+": {"value": 0}})
    sol = eq.solve(c, t_range=5, dt=0.001, tracker=None)
    np.testing.assert_allclose(sol.data, 2 - 2 * grid.axes_coords[0], rtol=5e-3)


def test_diffusion_cached(rng):
    """Test some caching of rhs of the simple diffusion model."""
    grid = UnitGrid([8])
    c0 = ScalarField.random_uniform(grid, rng=rng)

    # first run without cache
    eq1 = DiffusionPDE(diffusivity=1)
    eq1.cache_rhs = False
    c1a = eq1.solve(c0, t_range=1, dt=0.1, backend="numba", tracker=None)

    eq1.diffusivity = 0.1
    c1b = eq1.solve(c1a, t_range=1, dt=0.1, backend="numba", tracker=None)

    # then run with cache
    eq2 = DiffusionPDE(diffusivity=1)
    eq2.cache_rhs = True
    c2a = eq2.solve(c0, t_range=1, dt=0.1, backend="numba", tracker=None)

    eq2.diffusivity = 0.1
    c2b = eq2.solve(c2a, t_range=1, dt=0.1, backend="numba", tracker=None)

    eq2._cache = {}  # clear cache
    c2c = eq2.solve(c2a, t_range=1, dt=0.1, backend="numba", tracker=None)

    np.testing.assert_allclose(c1a.data, c2a.data)
    assert not np.allclose(c1b.data, c2b.data)
    np.testing.assert_allclose(c1b.data, c2c.data)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_diffusion_time_dependent_bcs(backend):
    """Test PDE with time-dependent BCs."""
    field = ScalarField(UnitGrid([3]))

    eq = DiffusionPDE(bc={"value_expression": "Heaviside(t - 1.5)"})

    storage = MemoryStorage()
    eq.solve(
        field,
        t_range=10,
        dt=1e-2,
        adaptive=True,
        backend=backend,
        tracker=storage.tracker(1),
    )

    np.testing.assert_allclose(storage[1].data, 0)
    np.testing.assert_allclose(storage[-1].data, 1, rtol=1e-3)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_diffusion_sde(backend, rng):
    """Test scaling of noise using a stochastic diffusion equation."""
    # we disable diffusivity to have a simple analytical solution
    var_local, t_range = 0.35, 0.1
    eq = DiffusionPDE(diffusivity=0, noise=var_local, rng=rng)
    grid = CartesianGrid([[0, 1000]], 3700)
    field = ScalarField(grid)
    sol = eq.solve(field, t_range=t_range, dt=1e-4, backend=backend, tracker=None)
    var_expected = var_local * t_range / grid.typical_discretization
    dist = stats.norm(scale=np.sqrt(var_expected)).cdf
    assert stats.kstest(np.ravel(sol.data), dist).pvalue > 0.1
