"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import random

import numpy as np
import pytest
from scipy import ndimage

from pde import CartesianGrid, ScalarField, Tensor2Field, UnitGrid, VectorField
from pde.grids.operators import cartesian as ops

π = np.pi


def _get_random_grid_bcs(ndim: int, dx="random", periodic="random", rank=0):
    """create a random Cartesian grid with natural bcs"""
    shape = tuple(np.random.randint(2, 5, ndim))

    if dx == "random":
        dx = np.random.uniform(0.5, 1.5, ndim)
    elif dx == "uniform":
        dx = np.full(ndim, np.random.uniform(0.5, 1.5))
    else:
        dx = np.broadcast_to(dx, shape)

    if periodic == "random":
        periodic = random.choice([True, False])

    sizes = [(0, float(s * d)) for s, d in zip(shape, dx)]
    grid = CartesianGrid(sizes, shape, periodic=periodic)
    return grid.get_boundary_conditions("auto_periodic_neumann", rank=rank)


@pytest.mark.parametrize("periodic", [True, False])
def test_singular_dimensions_2d(periodic):
    """test grids with singular dimensions"""
    dim = np.random.randint(3, 5)
    g1 = UnitGrid([dim], periodic=periodic)
    g2a = UnitGrid([dim, 1], periodic=periodic)
    g2b = UnitGrid([1, dim], periodic=periodic)

    field = ScalarField.random_uniform(g1)
    expected = field.laplace("auto_periodic_neumann").data
    for g in [g2a, g2b]:
        f = ScalarField(g, data=field.data.reshape(g.shape))
        res = f.laplace("auto_periodic_neumann").data.reshape(g1.shape)
        np.testing.assert_allclose(expected, res)


@pytest.mark.parametrize("periodic", [True, False])
def test_singular_dimensions_3d(periodic):
    """test grids with singular dimensions"""
    dim = np.random.randint(3, 5)
    g1 = UnitGrid([dim], periodic=periodic)
    g3a = UnitGrid([dim, 1, 1], periodic=periodic)
    g3b = UnitGrid([1, 1, dim], periodic=periodic)

    field = ScalarField.random_uniform(g1)
    expected = field.laplace("auto_periodic_neumann").data
    for g in [g3a, g3b]:
        f = ScalarField(g, data=field.data.reshape(g.shape))
        res = f.laplace("auto_periodic_neumann").data.reshape(g1.shape)
        np.testing.assert_allclose(expected, res)


def test_laplace_1d():
    """test the implementation of the laplace operator"""
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(1, periodic=periodic)
        field = ScalarField.random_uniform(bcs.grid)
        l1 = field.laplace(bcs, backend="scipy")
        l2 = field.laplace(bcs, backend="numba")
        np.testing.assert_allclose(l1.data, l2.data)


def test_laplace_2d():
    """test the implementation of the laplace operator"""
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(2, dx="uniform", periodic=periodic)
        a = np.random.random(bcs.grid.shape)  # test data

        dx = np.mean(bcs.grid.discretization)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / dx**2
        res = ndimage.convolve(a, kernel, mode="wrap" if periodic else "reflect")

        field = ScalarField(bcs.grid, data=a)
        l1 = field.laplace(bcs, backend="scipy")
        np.testing.assert_allclose(l1.data, res)

        l2 = field.laplace(bcs, backend="numba")
        np.testing.assert_allclose(l2.data, res)


def test_laplace_2d_nonuniform():
    """test the implementation of the laplace operator for
    non-uniform coordinates"""
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim=2, dx="random", periodic=periodic)

        dx = bcs.grid.discretization
        kernel_x = np.array([1, -2, 1]) / dx[0] ** 2
        kernel_y = np.array([1, -2, 1]) / dx[1] ** 2
        a = np.random.random(bcs.grid.shape)

        mode = "wrap" if periodic else "reflect"
        res = ndimage.convolve1d(a, kernel_x, axis=0, mode=mode)
        res += ndimage.convolve1d(a, kernel_y, axis=1, mode=mode)

        field = ScalarField(bcs.grid, data=a)
        lap = field.laplace(bcs, backend="numba")
        np.testing.assert_allclose(lap.data, res)


def test_laplace_3d():
    """test the implementation of the laplace operator"""
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim=3, dx="uniform", periodic=periodic)
        field = ScalarField.random_uniform(bcs.grid)
        l1 = field.laplace(bcs, backend="scipy")
        l2 = field.laplace(bcs, backend="numba")
        np.testing.assert_allclose(l1.data, l2.data)


def test_gradient_1d():
    """test specific boundary conditions for the 1d gradient"""
    grid = UnitGrid(5)

    b_l = {"type": "derivative", "value": -1}
    b_r = {"type": "derivative", "value": 1}
    bcs = grid.get_boundary_conditions([b_l, b_r])
    field = ScalarField(grid, np.arange(5))
    res = field.gradient(bcs)
    np.testing.assert_allclose(res.data, np.ones((1, 5)))

    b_l = {"type": "value", "value": 3}
    b_r = {"type": "value", "value": 3}
    bcs = grid.get_boundary_conditions([b_l, b_r])
    field = ScalarField(grid, np.full(5, 3))
    res = field.gradient(bcs)
    np.testing.assert_allclose(res.data, np.zeros((1, 5)))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_gradient_cart(ndim):
    """test different gradient operators"""
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic)
        field = ScalarField.random_uniform(bcs.grid)
        res1 = field.gradient(bcs, backend="scipy").data
        res2 = field.gradient(bcs, backend="numba").data
        assert res1.shape == (ndim,) + bcs.grid.shape
        np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_divergence_cart(ndim):
    """test different divergence operators"""
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic, rank=1)
        field = VectorField.random_uniform(bcs.grid)
        res1 = field.divergence(bcs, backend="scipy").data
        res2 = field.divergence(bcs, backend="numba").data
        np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_vector_gradient(ndim):
    """test different vector gradient operators"""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=1)
    field = VectorField.random_uniform(bcs.grid)
    res1 = field.gradient(bcs, backend="scipy").data
    res2 = field.gradient(bcs, backend="numba").data
    assert res1.shape == (ndim, ndim) + bcs.grid.shape
    np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_vector_laplace_cart(ndim):
    """test different vector laplace operators"""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=1)
    field = VectorField.random_uniform(bcs.grid)
    res1 = field.laplace(bcs, backend="scipy").data
    res2 = field.laplace(bcs, backend="numba").data
    assert res1.shape == (ndim,) + bcs.grid.shape
    np.testing.assert_allclose(res1, res2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_tensor_divergence_cart(ndim):
    """test different tensor divergence operators"""
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=2)
    field = Tensor2Field.random_uniform(bcs.grid)
    res1 = field.divergence(bcs, backend="scipy").data
    res2 = field.divergence(bcs, backend="numba").data
    assert res1.shape == (ndim,) + bcs.grid.shape
    np.testing.assert_allclose(res1, res2)


def test_div_grad_const():
    """compare div grad to laplace operator"""
    grid = CartesianGrid([[-1, 1]], 32)

    # test constant
    y = ScalarField(grid, 3)
    for bc in [{"type": "derivative", "value": 0}, {"type": "value", "value": 3}]:
        bcs = grid.get_boundary_conditions(bc)
        lap = y.laplace(bcs)
        divgrad = y.gradient(bcs).divergence("auto_periodic_curvature")
        np.testing.assert_allclose(lap.data, np.zeros(32))
        np.testing.assert_allclose(divgrad.data, np.zeros(32))


def test_div_grad_linear():
    """compare div grad to laplace operator"""
    grid = CartesianGrid([[-1, 1]], 32)
    x = grid.axes_coords[0]

    # test linear
    f = np.random.random() + 1
    y = ScalarField(grid, f * x)

    b1 = [{"type": "neumann", "value": -f}, {"type": "neumann", "value": f}]
    b2 = [{"type": "value", "value": -f}, {"type": "value", "value": f}]
    for bs in [b1, b2]:
        bcs = y.grid.get_boundary_conditions(bs)
        lap = y.laplace(bcs)
        divgrad = y.gradient(bcs).divergence("auto_periodic_curvature")
        np.testing.assert_allclose(lap.data, np.zeros(32), atol=1e-10)
        np.testing.assert_allclose(divgrad.data, np.zeros(32), atol=1e-10)


def test_div_grad_quadratic():
    """compare div grad to laplace operator"""
    grid = CartesianGrid([[-1, 1]], 32)
    x = grid.axes_coords[0]

    # test simple quadratic
    y = ScalarField(grid, x**2)

    bcs = grid.get_boundary_conditions({"type": "derivative", "value": 2})
    lap = y.laplace(bcs)
    divgrad = y.gradient(bcs).divergence("auto_periodic_curvature")

    np.testing.assert_allclose(lap.data, np.full(32, 2.0))
    np.testing.assert_allclose(divgrad.data, np.full(32, 2.0))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_gradient_squared_cart(dim):
    """compare gradient squared operator"""
    grid = CartesianGrid(
        [[0, 2 * np.pi]] * dim,
        shape=np.random.randint(30, 35, dim),
        periodic=np.random.choice([False, True], dim),
    )
    field = ScalarField.random_harmonic(grid, modes=1, axis_combination=np.add)
    s1 = field.gradient("auto_periodic_neumann").to_scalar("squared_sum")
    s2 = field.gradient_squared("auto_periodic_neumann", central=True)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.1, atol=0.1)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.2, atol=0.2)
    assert not np.array_equal(s2.data, s3.data)


def test_rect_div_grad():
    """compare div grad to laplacian"""
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    field = ScalarField(grid, data=np.cos(x) + np.sin(y))

    bcs = grid.get_boundary_conditions("auto_periodic_neumann")

    a = field.laplace(bcs)
    b = field.gradient(bcs).divergence("auto_periodic_curvature")
    np.testing.assert_allclose(a.data, -field.data, rtol=0.05, atol=0.01)
    np.testing.assert_allclose(b.data, -field.data, rtol=0.05, atol=0.01)


def test_degenerated_grid():
    """test operators on grids with singular dimensions"""
    g1 = CartesianGrid([[0, 1]], 4)
    g2 = CartesianGrid([[0, 1], [0, 0.1]], [4, 1], periodic=[False, True])
    f1 = ScalarField.random_uniform(g1)
    f2 = ScalarField(g2, f1.data.reshape(g2.shape))

    res1 = f1.laplace("auto_periodic_neumann").data
    res2 = f2.laplace("auto_periodic_neumann").data
    np.testing.assert_allclose(res1.flat, res2.flat)


def test_2nd_order_bc():
    """test whether 2nd order boundary conditions can be used"""
    grid = UnitGrid([8, 8])
    field = ScalarField.random_uniform(grid)
    field.laplace([{"value": "sin(y)"}, {"value": "x"}])


@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_make_derivative(ndim, axis):
    """test the _make_derivative function"""
    periodic = random.choice([True, False])
    grid = CartesianGrid([[0, 6 * np.pi]] * ndim, 16, periodic=periodic)
    field = ScalarField.random_harmonic(grid, modes=1, axis_combination=np.add)

    bcs = grid.get_boundary_conditions("auto_periodic_neumann")
    grad = field.gradient(bcs)
    for method in ["central", "forward", "backward"]:
        msg = f"method={method}, periodic={periodic}"
        diff = ops._make_derivative(grid, axis=axis, method=method)
        res = field.copy()
        res.data[:] = 0
        field.set_ghost_cells(bcs)
        diff(field._data_full, out=res.data)
        np.testing.assert_allclose(
            grad.data[axis], res.data, atol=0.1, rtol=0.1, err_msg=msg
        )


@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_make_derivative2(ndim, axis):
    """test the _make_derivative2 function"""
    periodic = random.choice([True, False])
    grid = CartesianGrid([[0, 6 * np.pi]] * ndim, 16, periodic=periodic)
    field = ScalarField.random_harmonic(grid, modes=1, axis_combination=np.add)

    bcs = grid.get_boundary_conditions("auto_periodic_neumann")
    grad = field.gradient(bcs)[axis]
    grad2 = grad.gradient(bcs)[axis]

    diff = ops._make_derivative2(grid, axis=axis)
    res = field.copy()
    res.data[:] = 0
    field.set_ghost_cells(bcs)
    diff(field._data_full, out=res.data)
    np.testing.assert_allclose(grad2.data, res.data, atol=0.1, rtol=0.1)
