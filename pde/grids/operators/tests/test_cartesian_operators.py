"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import random

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField, UnitGrid
from pde.grids.operators import cartesian as ops
from scipy import ndimage

π = np.pi


def _get_random_grid_bcs(ndim: int, dx="random", periodic="random", rank=0):
    """ create a random Cartesian grid with natural bcs """
    shape = np.random.randint(2, 5, ndim)

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
    return grid.get_boundary_conditions("natural", rank=rank)


@pytest.mark.parametrize("periodic", [True, False])
def test_singular_dimensions_2d(periodic):
    """ test grids with singular dimensions """
    dim = np.random.randint(3, 5)
    g1 = UnitGrid([dim], periodic=periodic)
    g2a = UnitGrid([dim, 1], periodic=periodic)
    g2b = UnitGrid([1, dim], periodic=periodic)

    data = np.random.random(dim)
    expected = g1.get_operator("laplace", "natural")(data)
    for g in [g2a, g2b]:
        res = g.get_operator("laplace", "natural")(data.reshape(g.shape))
        np.testing.assert_allclose(expected.flat, res.flat)


@pytest.mark.parametrize("periodic", [True, False])
def test_singular_dimensions_3d(periodic):
    """ test grids with singular dimensions """
    dim = np.random.randint(3, 5)
    g1 = UnitGrid([dim], periodic=periodic)
    g3a = UnitGrid([dim, 1, 1], periodic=periodic)
    g3b = UnitGrid([1, 1, dim], periodic=periodic)

    data = np.random.random(dim)
    expected = g1.get_operator("laplace", "natural")(data)
    for g in [g3a, g3b]:
        res = g.get_operator("laplace", "natural")(data.reshape(g.shape))
        np.testing.assert_allclose(expected.flat, res.flat)


def test_laplace_1d():
    """ test the implementation of the laplace operator """
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(1, periodic=periodic)
        a = np.random.random(bcs.grid.shape)  # test data
        l1 = ops.make_laplace(bcs, method="scipy")
        l2 = ops.make_laplace(bcs, method="numba")
        l3 = ops.make_laplace(bcs, method="matrix")
        np.testing.assert_allclose(l1(a), l2(a))
        np.testing.assert_allclose(l1(a), l3(a))


def test_laplace_2d():
    """ test the implementation of the laplace operator """
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(2, dx="uniform", periodic=periodic)
        a = np.random.random(bcs.grid.shape)  # test data

        dx = bcs._uniform_discretization
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / dx ** 2
        res = ndimage.convolve(a, kernel, **bcs._scipy_border_mode)
        l1 = ops.make_laplace(bcs, method="scipy")
        np.testing.assert_allclose(l1(a), res)

        l2 = ops.make_laplace(bcs, method="numba")
        np.testing.assert_allclose(l2(a), res)

        l3 = ops.make_laplace(bcs, method="matrix")
        np.testing.assert_allclose(l3(a), res)


def test_laplace_2d_nonuniform():
    """test the implementation of the laplace operator for
    non-uniform coordinates"""
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim=2, dx="random", periodic=periodic)

        dx = bcs.grid.discretization
        kernel_x = np.array([1, -2, 1]) / dx[0] ** 2
        kernel_y = np.array([1, -2, 1]) / dx[1] ** 2
        a = np.random.random(bcs.grid.shape)

        res = ndimage.convolve1d(a, kernel_x, axis=0, **bcs._scipy_border_mode)
        res += ndimage.convolve1d(a, kernel_y, axis=1, **bcs._scipy_border_mode)

        lap = ops._make_laplace_numba_2d(bcs)
        np.testing.assert_allclose(lap(a), res)

        lap = ops.make_laplace(bcs, method="matrix")
        np.testing.assert_allclose(lap(a), res)


def test_laplace_3d():
    """ test the implementation of the laplace operator """
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim=3, dx="uniform", periodic=periodic)
        a = np.random.random(bcs.grid.shape)
        l1 = ops.make_laplace(bcs, method="scipy")
        l2 = ops.make_laplace(bcs, method="numba")
        np.testing.assert_allclose(l1(a), l2(a))


def test_gradient_1d():
    """ test specific boundary conditions for the 1d gradient """
    grid = UnitGrid(5)

    b_l = {"type": "derivative", "value": -1}
    b_r = {"type": "derivative", "value": 1}
    bcs = grid.get_boundary_conditions([b_l, b_r])
    grad = ops._make_gradient_numba_1d(bcs)
    np.testing.assert_allclose(grad(np.arange(5)), np.ones((1, 5)))

    b_l = {"type": "value", "value": 3}
    b_r = {"type": "value", "value": 3}
    bcs = grid.get_boundary_conditions([b_l, b_r])
    grad = ops._make_gradient_numba_1d(bcs)
    np.testing.assert_allclose(grad(np.full(5, 3)), np.zeros((1, 5)))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_gradient(ndim):
    """ test different gradient operators """
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic)
        op1 = ops.make_gradient(bcs, method="scipy")
        op2 = ops.make_gradient(bcs, method="numba")
        arr = np.random.random(bcs.grid.shape)
        val1, val2 = op1(arr), op2(arr)
        assert val1.shape == (ndim,) + bcs.grid.shape
        np.testing.assert_allclose(val1, val2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_divergence(ndim):
    """ test different divergence operators """
    for periodic in [True, False]:
        bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic=periodic)
        op1 = ops.make_divergence(bcs, method="scipy")
        op2 = ops.make_divergence(bcs, method="numba")
        arr = np.random.random(np.r_[ndim, bcs.grid.shape])
        np.testing.assert_allclose(op1(arr), op2(arr))


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_vector_gradient(ndim):
    """ test different vector gradient operators """
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=1)
    mvg = ops.make_vector_gradient
    op1 = mvg(bcs, method="scipy")
    op2 = mvg(bcs, method="numba")
    arr = np.random.random((ndim,) + bcs.grid.shape)
    val1, val2 = op1(arr), op2(arr)
    assert val1.shape == (ndim, ndim) + bcs.grid.shape
    np.testing.assert_allclose(val1, val2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_vector_laplace(ndim):
    """ test different vector laplace operators """
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=1)
    mvg = ops.make_vector_laplace
    op1 = mvg(bcs, method="scipy")
    op2 = mvg(bcs, method="numba")
    arr = np.random.random((ndim,) + bcs.grid.shape)
    val1, val2 = op1(arr), op2(arr)
    assert val1.shape == (ndim,) + bcs.grid.shape
    np.testing.assert_allclose(val1, val2)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_tensor_divergence(ndim):
    """ test different tensor divergence operators """
    bcs = _get_random_grid_bcs(ndim, dx="uniform", periodic="random", rank=1)
    op1 = ops.make_tensor_divergence(bcs, method="scipy")
    op2 = ops.make_tensor_divergence(bcs, method="numba")
    arr = np.random.random((ndim, ndim) + bcs.grid.shape)
    val1, val2 = op1(arr), op2(arr)
    assert val1.shape == (ndim,) + bcs.grid.shape
    np.testing.assert_allclose(val1, val2)


def test_div_grad_const():
    """ compare div grad to laplace operator """
    grid = CartesianGrid([[-1, 1]], 32)

    # test constant
    y = ScalarField(grid, 3)
    for bc in [{"type": "derivative", "value": 0}, {"type": "value", "value": 3}]:
        bcs = grid.get_boundary_conditions(bc)
        lap = y.laplace(bcs)
        divgrad = y.gradient(bcs).divergence(bcs.differentiated)
        np.testing.assert_allclose(lap.data, np.zeros(32))
        np.testing.assert_allclose(divgrad.data, np.zeros(32))


def test_div_grad_linear():
    """ compare div grad to laplace operator """
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
        divgrad = y.gradient(bcs).divergence(bcs.differentiated)
        np.testing.assert_allclose(lap.data, np.zeros(32), atol=1e-10)
        np.testing.assert_allclose(divgrad.data, np.zeros(32), atol=1e-10)


def test_div_grad_quadratic():
    """ compare div grad to laplace operator """
    grid = CartesianGrid([[-1, 1]], 32)
    x = grid.axes_coords[0]

    # test simple quadratic
    y = ScalarField(grid, x ** 2)

    bcs = grid.get_boundary_conditions({"type": "derivative", "value": 2})
    lap = y.laplace(bcs)
    divgrad = y.gradient(bcs).divergence(bcs.differentiated)

    np.testing.assert_allclose(lap.data, np.full(32, 2.0))
    np.testing.assert_allclose(divgrad.data, np.full(32, 2.0))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_gradient_squared(dim):
    """ compare gradient squared operator """
    grid = CartesianGrid(
        [[0, 2 * np.pi]] * dim,
        shape=np.random.randint(30, 35, dim),
        periodic=np.random.choice([False, True], dim),
    )
    field = ScalarField.random_harmonic(grid, modes=1)
    s1 = field.gradient("natural").to_scalar("squared_sum")
    s2 = field.gradient_squared("natural", central=True)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.1, atol=0.1)
    s3 = field.gradient_squared("natural", central=False)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.2, atol=0.2)
    assert not np.array_equal(s2.data, s3.data)


def test_rect_div_grad():
    """ compare div grad to laplacian """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    arr = np.cos(x) + np.sin(y)

    bcs = grid.get_boundary_conditions("natural")
    laplace = grid.get_operator("laplace", bcs)
    grad = grid.get_operator("gradient", bcs)
    div = grid.get_operator("divergence", bcs.differentiated)
    a = laplace(arr)
    b = div(grad(arr))
    np.testing.assert_allclose(a, -arr, rtol=0.05, atol=0.01)
    np.testing.assert_allclose(b, -arr, rtol=0.05, atol=0.01)


def test_degenerated_grid():
    """ test operators on grids with singular dimensions """
    g1 = CartesianGrid([[0, 1]], 4)
    g2 = CartesianGrid([[0, 1], [0, 0.1]], [4, 1], periodic=[False, True])
    d = np.random.random(4)

    v1 = g1.get_operator("laplace", bc="natural")(d)
    v2 = g2.get_operator("laplace", bc="natural")(d.reshape(g2.shape))
    assert v2.shape == g2.shape

    np.testing.assert_allclose(v1.flat, v2.flat)


@pytest.mark.parametrize("dim", [1, 2])
def test_poisson_solver_general(dim):
    """ test the poisson solver on Cartesian grids """
    bcs = _get_random_grid_bcs(dim)

    poisson = bcs.grid.get_operator("poisson_solver", bcs)
    laplace = bcs.grid.get_operator("laplace", bcs)

    d = np.random.random(bcs.grid.shape)
    d -= d.mean()  # balance the right hand side
    np.testing.assert_allclose(
        laplace(poisson(d)), d, atol=1e-5, rtol=1e-5, err_msg=f"bcs = {bcs}"
    )


def test_2nd_order_bc():
    """ test whether 2nd order boundary conditions can be used """
    grid = UnitGrid([8, 8])
    field = ScalarField.random_uniform(grid)
    field.laplace([{"value": "sin(y)"}, {"value": "x"}])


@pytest.mark.parametrize("ndim,axis", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_make_derivative(ndim, axis):
    """ test the make derivative function """
    periodic = random.choice([True, False])
    grid = CartesianGrid([[0, 6 * np.pi]] * ndim, 16, periodic=periodic)
    field = ScalarField.random_harmonic(grid, modes=1, axis_combination=np.add)

    bcs = grid.get_boundary_conditions("natural")
    grad = field.gradient(bcs)
    for method in ["central", "forward", "backward"]:
        msg = f"method={method}, periodic={periodic}"
        diff = ops._make_derivative(bcs, axis=axis, method=method)
        np.testing.assert_allclose(
            grad.data[axis], diff(field.data), atol=0.1, rtol=0.1, err_msg=msg
        )
