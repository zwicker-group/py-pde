"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import random

import numpy as np
import pytest
from scipy import ndimage

from pde import CartesianGrid, ScalarField, Tensor2Field, UnitGrid, VectorField

pytest.importorskip("torch")

π = np.pi


def _get_random_grid_bcs(ndim: int, dx="random", periodic="random", rank=0):
    """Create a random Cartesian grid with auto_periodic_neumann bcs."""
    rng = np.random.default_rng(0)
    shape = tuple(rng.integers(2, 5, ndim))

    if dx == "random":
        dx = rng.uniform(0.5, 1.5, ndim)
    elif dx == "uniform":
        dx = np.full(ndim, rng.uniform(0.5, 1.5))
    else:
        dx = np.broadcast_to(dx, shape)

    if periodic == "random":
        periodic = random.choice([True, False])

    sizes = [(0, float(s * d)) for s, d in zip(shape, dx, strict=False)]
    grid = CartesianGrid(sizes, shape, periodic=periodic)
    return grid.get_boundary_conditions("auto_periodic_neumann", rank=rank)


@pytest.mark.parametrize("periodic", [True, False])
def test_singular_dimensions_2d(periodic, rng):
    """Test grids with singular dimensions."""
    dim = rng.integers(3, 5)
    g1 = UnitGrid([dim], periodic=periodic)
    g2a = UnitGrid([dim, 1], periodic=periodic)
    g2b = UnitGrid([1, dim], periodic=periodic)

    field = ScalarField.random_uniform(g1, rng=rng)
    expected = field.laplace("auto_periodic_neumann", backend="torch").data
    for g in [g2a, g2b]:
        f = ScalarField(g, data=field.data.reshape(g.shape))
        res = f.laplace("auto_periodic_neumann", backend="torch").data.reshape(g1.shape)
        np.testing.assert_allclose(expected, res)


@pytest.mark.parametrize("periodic", [True, False])
def test_laplace_1d(periodic, rng):
    """Test the implementation of the laplace operator."""
    bcs = _get_random_grid_bcs(1, periodic=periodic)
    field = ScalarField.random_colored(bcs.grid, -6, rng=rng)
    l1 = field.laplace(bcs, backend="scipy")
    l2 = field.laplace(bcs, backend="torch")
    np.testing.assert_allclose(l1.data, l2.data)


@pytest.mark.parametrize("periodic", [True, False])
def test_laplace_2d_nonuniform(periodic, rng):
    """Test the implementation of the laplace operator for non-uniform coordinates."""
    bcs = _get_random_grid_bcs(ndim=2, dx="random", periodic=periodic)

    dx = bcs.grid.discretization
    kernel_x = np.array([1, -2, 1]) / dx[0] ** 2
    kernel_y = np.array([1, -2, 1]) / dx[1] ** 2
    a = rng.random(bcs.grid.shape)

    mode = "wrap" if periodic else "reflect"
    res = ndimage.convolve1d(a, kernel_x, axis=0, mode=mode)
    res += ndimage.convolve1d(a, kernel_y, axis=1, mode=mode)

    field = ScalarField(bcs.grid, data=a)
    lap = field.laplace(bcs, backend="torch")
    np.testing.assert_allclose(lap.data, res)


@pytest.mark.parametrize("periodic", [True, False])
def test_laplace_3d(periodic, rng):
    """Test the implementation of the laplace operator."""
    bcs = _get_random_grid_bcs(ndim=3, dx="uniform", periodic=periodic)
    field = ScalarField.random_uniform(bcs.grid, rng=rng)
    l1 = field.laplace(bcs, backend="scipy")
    l2 = field.laplace(bcs, backend="torch")
    np.testing.assert_allclose(l1.data, l2.data)


def test_gradient_1d():
    """Test specific boundary conditions for the 1d gradient."""
    grid = UnitGrid(5)

    bc = {"x-": {"derivative": -1}, "x+": {"derivative": 1}}
    bcs = grid.get_boundary_conditions(bc)
    field = ScalarField(grid, np.arange(5))
    res = field.gradient(bcs, backend="torch")
    np.testing.assert_allclose(res.data, np.ones((1, 5)))

    bcs = grid.get_boundary_conditions({"x": {"value": 3}})
    field = ScalarField(grid, np.full(5, 3))
    res = field.gradient(bcs, backend="torch")
    np.testing.assert_allclose(res.data, np.zeros((1, 5)))


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_gradient_cart(ndim, periodic, rng):
    """Test different gradient operators."""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic)
    field = ScalarField.random_uniform(bcs.grid, rng=rng)
    res1 = field.gradient(bcs, backend="scipy").data
    res2 = field.gradient(bcs, backend="torch").data
    assert res1.shape == (ndim, *bcs.grid.shape)
    np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_gradient_squared_cart(dim, rng):
    """Compare gradient squared operator."""
    grid = CartesianGrid(
        [[0, 2 * np.pi]] * dim,
        shape=rng.integers(30, 35, dim),
        periodic=rng.choice([False, True], dim),
    )
    field = ScalarField.random_harmonic(grid, modes=1, axis_combination=np.add, rng=rng)
    s1 = field.gradient("auto_periodic_neumann").to_scalar("squared_sum")
    s2 = field.gradient_squared("auto_periodic_neumann", central=True, backend="torch")
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.1, atol=0.1)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False, backend="torch")
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.2, atol=0.2)
    assert not np.array_equal(s2.data, s3.data)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_divergence_cart(ndim, periodic, rng):
    """Test different divergence operators."""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic, rank=1)
    field = VectorField.random_uniform(bcs.grid, rng=rng)
    res1 = field.divergence(bcs, backend="scipy").data
    res2 = field.divergence(bcs, backend="torch").data
    np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_vector_gradient(ndim, rng):
    """Test different vector gradient operators."""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=1)
    field = VectorField.random_uniform(bcs.grid, rng=rng)
    res1 = field.gradient(bcs, backend="scipy").data
    res2 = field.gradient(bcs, backend="torch").data
    assert res1.shape == (ndim, ndim, *bcs.grid.shape)
    np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_vector_laplace_cart(ndim, rng):
    """Test different vector laplace operators."""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=1)
    field = VectorField.random_uniform(bcs.grid, rng=rng)
    res1 = field.laplace(bcs, backend="scipy").data
    res2 = field.laplace(bcs, backend="torch").data
    assert res1.shape == (ndim, *bcs.grid.shape)
    np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_tensor_divergence_cart(ndim, rng):
    """Test different tensor divergence operators."""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=2)
    field = Tensor2Field.random_uniform(bcs.grid, rng=rng)
    res1 = field.divergence(bcs, backend="scipy").data
    res2 = field.divergence(bcs, backend="torch").data
    assert res1.shape == (ndim, *bcs.grid.shape)
    np.testing.assert_allclose(res1, res2)


def test_2nd_order_bc(rng):
    """Test whether 2nd order boundary conditions can be used."""
    grid = UnitGrid([8, 8])
    field = ScalarField.random_uniform(grid, rng=rng)
    field.laplace({"x": {"value": "sin(y)"}, "y": {"value": "x"}}, backend="torch")
