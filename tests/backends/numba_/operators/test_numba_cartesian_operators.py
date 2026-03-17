"""Tests for numba-specific Cartesian grid operator features.

These tests cover functionality that is unique to the numba backend, such as
forward/backward finite-difference methods, spectral operators, corner-point
stencils, and degenerated-grid handling. Generic operator tests that are shared
across backends live in ``tests/backends/generic/operators/``.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import random

import numpy as np
import pytest
from scipy import ndimage

from pde import CartesianGrid, ScalarField, UnitGrid, VectorField
from pde.backends.numba.operators import cartesian as ops
from pde.tools.misc import module_available


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
def test_singular_dimensions_3d(periodic, rng):
    """Test grids with singular dimensions."""
    dim = rng.integers(3, 5)
    g1 = UnitGrid([dim], periodic=periodic)
    g3a = UnitGrid([dim, 1, 1], periodic=periodic)
    g3b = UnitGrid([1, 1, dim], periodic=periodic)

    field = ScalarField.random_uniform(g1, rng=rng)
    expected = field.laplace("auto_periodic_neumann", backend="numba").data
    for g in [g3a, g3b]:
        f = ScalarField(g, data=field.data.reshape(g.shape))
        res = f.laplace("auto_periodic_neumann", backend="numba").data.reshape(g1.shape)
        np.testing.assert_allclose(expected, res)


@pytest.mark.skipif(not module_available("rocket_fft"), reason="requires `rocket_fft`")
@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("dtype", [float, complex])
def test_laplace_spectral(ndim, dtype, rng):
    """Test the implementation of the spectral laplace operator."""
    shape = np.c_[rng.uniform(-20, -10, ndim), rng.uniform(10, 20, ndim)]
    grid = CartesianGrid(shape, 30, periodic=True)
    std = 1 if dtype is float else 1 + 1j
    field = ScalarField.random_normal(
        grid, std=std, correlation="gaussian", length_scale=20, dtype=dtype, rng=rng
    )
    field /= np.real(field).fluctuations
    l1 = field.laplace("periodic", backend="numba", spectral=False)
    l2 = field.laplace("periodic", backend="numba", spectral=True)
    np.testing.assert_allclose(l1.data, l2.data, atol=0.1, rtol=0.01)


@pytest.mark.parametrize("periodic", [True, False])
def test_laplace_2d(periodic, rng):
    """Test the implementation of the laplace operator."""
    bcs = _get_random_grid_bcs(2, dx="uniform", periodic=periodic)
    a = rng.random(bcs.grid.shape)  # test data

    dx = np.mean(bcs.grid.discretization)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / dx**2
    res = ndimage.convolve(a, kernel, mode="wrap" if periodic else "reflect")

    field = ScalarField(bcs.grid, data=a)
    l1 = field.laplace(bcs, backend="scipy")
    np.testing.assert_allclose(l1.data, res)

    l2 = field.laplace(bcs, backend="numba")
    np.testing.assert_allclose(l2.data, res)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("method", ["central", "forward", "backward"])
@pytest.mark.parametrize("periodic", [True, False])
def test_gradient_cart_methods(ndim, method, periodic, rng):
    """Test gradient operators with forward and backward finite-difference methods."""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic)
    field = ScalarField.random_uniform(bcs.grid, rng=rng)
    res1 = field.gradient(bcs, backend="scipy", method=method).data
    res2 = field.gradient(bcs, backend="numba", method=method).data
    assert res1.shape == (ndim, *bcs.grid.shape)
    np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("method", ["central", "forward", "backward"])
@pytest.mark.parametrize("periodic", [True, False])
def test_divergence_cart_methods(ndim, method, periodic, rng):
    """Test divergence operators with forward and backward finite-difference methods."""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic, rank=1)
    field = VectorField.random_uniform(bcs.grid, rng=rng)
    res1 = field.divergence(bcs, backend="scipy", method=method).data
    res2 = field.divergence(bcs, backend="numba", method=method).data
    np.testing.assert_allclose(res1, res2)


def test_div_grad_const():
    """Compare div grad to laplace operator."""
    grid = CartesianGrid([[-1, 1]], 32)

    # test constant
    y = ScalarField(grid, 3)
    for bc in [{"type": "derivative", "value": 0}, {"type": "value", "value": 3}]:
        bcs = grid.get_boundary_conditions(bc)
        lap = y.laplace(bcs, backend="numba")
        divgrad = y.gradient(bcs, backend="numba").divergence(
            "auto_periodic_curvature", backend="numba"
        )
        np.testing.assert_allclose(lap.data, np.zeros(32))
        np.testing.assert_allclose(divgrad.data, np.zeros(32))


def test_div_grad_linear(rng):
    """Compare div grad to laplace operator."""
    grid = CartesianGrid([[-1, 1]], 32)
    x = grid.axes_coords[0]

    # test linear
    f = rng.random() + 1
    y = ScalarField(grid, f * x)

    b1 = {"x-": {"derivative": -f}, "x+": {"derivative": f}}
    b2 = {"x-": {"value": -f}, "x+": {"value": f}}
    for bs in [b1, b2]:
        bcs = y.grid.get_boundary_conditions(bs)
        lap = y.laplace(bcs, backend="numba")
        divgrad = y.gradient(bcs, backend="numba").divergence(
            "auto_periodic_curvature", backend="numba"
        )
        np.testing.assert_allclose(lap.data, np.zeros(32), atol=1e-10)
        np.testing.assert_allclose(divgrad.data, np.zeros(32), atol=1e-10)


def test_div_grad_quadratic():
    """Compare div grad to laplace operator."""
    grid = CartesianGrid([[-1, 1]], 32)
    x = grid.axes_coords[0]

    # test simple quadratic
    y = ScalarField(grid, x**2)

    bcs = grid.get_boundary_conditions({"type": "derivative", "value": 2})
    lap = y.laplace(bcs, backend="numba")
    divgrad = y.gradient(bcs, backend="numba").divergence(
        "auto_periodic_curvature", backend="numba"
    )

    np.testing.assert_allclose(lap.data, np.full(32, 2.0))
    np.testing.assert_allclose(divgrad.data, np.full(32, 2.0))


def test_rect_div_grad():
    """Compare div grad to laplacian."""
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    field = ScalarField(grid, data=np.cos(x) + np.sin(y))

    bcs = grid.get_boundary_conditions("auto_periodic_neumann")

    a = field.laplace(bcs, backend="numba")
    b = field.gradient(bcs, backend="numba").divergence(
        "auto_periodic_curvature", backend="numba"
    )
    np.testing.assert_allclose(a.data, -field.data, rtol=0.05, atol=0.01)
    np.testing.assert_allclose(b.data, -field.data, rtol=0.05, atol=0.01)


def test_degenerated_grid(rng):
    """Test operators on grids with singular dimensions."""
    g1 = CartesianGrid([[0, 1]], 4)
    g2 = CartesianGrid([[0, 1], [0, 0.1]], [4, 1], periodic=[False, True])
    f1 = ScalarField.random_uniform(g1, rng=rng)
    f2 = ScalarField(g2, f1.data.reshape(g2.shape))

    res1 = f1.laplace("auto_periodic_neumann", backend="numba").data
    res2 = f2.laplace("auto_periodic_neumann", backend="numba").data
    np.testing.assert_allclose(res1.flat, res2.flat)


@pytest.mark.parametrize("periodic_x", [False, True])
@pytest.mark.parametrize("periodic_y", [False, True])
def test_corner_point_setter(periodic_x, periodic_y):
    """Test the corner point setter."""
    grid = UnitGrid([1, 1], periodic=[periodic_x, periodic_y])

    arr = np.array([[np.nan, 1, np.nan], [2, 3, 4], [np.nan, 5, np.nan]])
    if periodic_x:
        arr[0, :] = arr[2, :] = arr[1, :]
    if periodic_y:
        arr[:, 0] = arr[:, 2] = arr[:, 1]

    setter = ops.make_corner_point_setter_2d(grid)
    setter(arr)

    if periodic_x and periodic_y:
        np.testing.assert_allclose(arr, 3)
    elif periodic_x and not periodic_y:
        np.testing.assert_allclose(arr, [[2, 3, 4], [2, 3, 4], [2, 3, 4]])
    elif not periodic_x and periodic_y:
        np.testing.assert_allclose(arr, [[1, 1, 1], [3, 3, 3], [5, 5, 5]])
    elif not periodic_x and not periodic_y:
        np.testing.assert_allclose(2 * arr, [[3, 2, 5], [4, 6, 8], [7, 10, 9]])


@pytest.mark.parametrize("periodic_x", [False, True])
@pytest.mark.parametrize("periodic_y", [False, True])
def test_9point_stencil(periodic_x, periodic_y, rng):
    """Test the corner point setter."""
    grid = UnitGrid([16, 16], periodic=[periodic_x, periodic_y])
    field = ScalarField.random_uniform(grid, rng=rng).smooth(1)
    lap1 = field.laplace(bc="auto_periodic_neumann", backend="numba")

    lap2 = field.laplace(
        bc="auto_periodic_neumann", corner_weight=1e-10, backend="numba"
    )
    np.testing.assert_allclose(lap1.data, lap2.data)

    lap3 = field.laplace(
        bc="auto_periodic_neumann", corner_weight=1 / 3, backend="numba"
    )
    np.testing.assert_allclose(lap1.data, lap3.data, atol=0.05)
