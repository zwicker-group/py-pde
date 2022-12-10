"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import (
    CartesianGrid,
    PolarSymGrid,
    ScalarField,
    Tensor2Field,
    VectorField,
    solve_poisson_equation,
)


def test_findiff_polar():
    """test operator for a simple polar grid"""
    grid = PolarSymGrid(1.5, 3)
    _, _, r2 = grid.axes_coords[0]
    assert grid.discretization == (0.5,)
    s = ScalarField(grid, [1, 2, 4])
    v = VectorField(grid, [[1, 2, 4], [0] * 3])

    # test gradient
    grad = s.gradient(bc=["derivative", "value"])
    np.testing.assert_allclose(grad.data[0, :], [1, 3, -6])
    grad = s.gradient(bc="derivative")
    np.testing.assert_allclose(grad.data[0, :], [1, 3, 2])

    # test divergence
    div = v.divergence(bc=["derivative", "value"])
    np.testing.assert_allclose(div.data, [5, 17 / 3, -6 + 4 / r2])
    div = v.divergence(bc="derivative")
    np.testing.assert_allclose(div.data, [5, 17 / 3, 2 + 4 / r2])


def test_conservative_laplace_polar():
    """test and compare the two implementation of the laplace operator"""
    grid = PolarSymGrid(1.5, 8)
    f = ScalarField.random_uniform(grid)

    res = f.laplace("auto_periodic_neumann")
    np.testing.assert_allclose(res.integral, 0, atol=1e-12)


@pytest.mark.parametrize(
    "op_name,field",
    [
        ("laplace", ScalarField),
        ("divergence", VectorField),
        ("gradient", ScalarField),
        ("tensor_divergence", Tensor2Field),
    ],
)
def test_small_annulus_polar(op_name, field):
    """test whether a small annulus gives the same result as a sphere"""
    grids = [
        PolarSymGrid((0, 1), 8),
        PolarSymGrid((1e-8, 1), 8),
        PolarSymGrid((0.1, 1), 8),
    ]

    f = field.random_uniform(grids[0])

    res = [
        field(g, data=f.data)._apply_operator(op_name, "auto_periodic_neumann")
        for g in grids
    ]

    np.testing.assert_almost_equal(res[0].data, res[1].data, decimal=5)
    assert np.linalg.norm(res[0].data - res[2].data) > 1e-3


def test_grid_laplace_polar():
    """test the polar implementation of the laplace operator"""
    grid_sph = PolarSymGrid(7, 8)
    grid_cart = CartesianGrid([[-5, 5], [-5, 5]], [12, 11])

    a_1d = ScalarField.from_expression(grid_sph, "cos(r)")
    a_2d = a_1d.interpolate_to_grid(grid_cart)

    b_2d = a_2d.laplace("auto_periodic_neumann")
    b_1d = a_1d.laplace("auto_periodic_neumann")
    b_1d_2 = b_1d.interpolate_to_grid(grid_cart)

    i = slice(1, -1)  # do not compare boundary points
    np.testing.assert_allclose(b_1d_2.data[i, i], b_2d.data[i, i], rtol=0.2, atol=0.2)


@pytest.mark.parametrize("r_inner", (0, 2 * np.pi))
def test_gradient_squared_polar(r_inner):
    """compare gradient squared operator"""
    grid = PolarSymGrid((r_inner, 4 * np.pi), 32)
    field = ScalarField.from_expression(grid, "cos(r)")
    s1 = field.gradient("auto_periodic_neumann").to_scalar("squared_sum")
    s2 = field.gradient_squared("auto_periodic_neumann", central=True)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.1, atol=0.1)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.1, atol=0.1)
    assert not np.array_equal(s2.data, s3.data)


def test_grid_div_grad_polar():
    """compare div grad to laplacian for polar grids"""
    grid = PolarSymGrid(2 * np.pi, 16)
    field = ScalarField.from_expression(grid, "cos(r)")

    a = field.laplace("derivative")
    b = field.gradient("derivative").divergence("value")
    res = ScalarField.from_expression(grid, "-sin(r) / r - cos(r)")

    # do not test the radial boundary points
    np.testing.assert_allclose(a.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)
    np.testing.assert_allclose(b.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)


def test_poisson_solver_polar():
    """test the poisson solver on Polar grids"""
    for grid in [PolarSymGrid(4, 8), PolarSymGrid([2, 4], 8)]:
        for bc_val in ["auto_periodic_neumann", {"value": 1}]:
            bcs = grid.get_boundary_conditions(bc_val)
            d = ScalarField.random_uniform(grid)
            d -= d.average  # balance the right hand side
            sol = solve_poisson_equation(d, bcs)
            test = sol.laplace(bcs)
            msg = f"grid={grid}, bcs={bc_val}"
            np.testing.assert_allclose(test.data, d.data, err_msg=msg, rtol=1e-6)


def test_examples_scalar_polar():
    """compare derivatives of scalar fields for polar grids"""
    grid = PolarSymGrid(1, 32)
    sf = ScalarField.from_expression(grid, "r**3")

    # gradient
    res = sf.gradient([{"derivative": 0}, {"derivative": 3}])
    expect = VectorField.from_expression(grid, ["3 * r**2", 0])
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # gradient squared
    expect = ScalarField.from_expression(grid, "9 * r**4")
    for c in [True, False]:
        res = sf.gradient_squared([{"derivative": 0}, {"derivative": 3}], central=c)
        np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # laplace
    res = sf.laplace([{"derivative": 0}, {"derivative": 3}])
    expect = ScalarField.from_expression(grid, "9 * r")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_vector_polar():
    """compare derivatives of vector fields for polar grids"""
    grid = PolarSymGrid(1, 32)
    vf = VectorField.from_expression(grid, ["r**3", "r**2"])

    # divergence
    res = vf.divergence([{"derivative": 0}, {"value": 1}])
    expect = ScalarField.from_expression(grid, "4 * r**2")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # # vector Laplacian
    # res = vf.laplace([{"derivative": 0}, {"value": 1}])
    # expect = VectorField.from_expression(grid, ["8 * r", "3"])
    # np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # vector gradient
    res = vf.gradient([{"derivative": 0}, {"value": [1, 1]}])
    expr = [["3 * r**2", "-r"], ["2 * r", "r**2"]]
    expect = Tensor2Field.from_expression(grid, expr)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_tensor_polar():
    """compare derivatives of tensorial fields for polar grids"""
    grid = PolarSymGrid(1, 32)
    tf = Tensor2Field.from_expression(grid, [["r**3"] * 2] * 2)

    # tensor divergence
    res = tf.divergence([{"derivative": 0}, {"normal_value": [1, 1]}])
    expect = VectorField.from_expression(grid, ["3 * r**2", "5 * r**2"])
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    res = tf.divergence([{"derivative": 0}, {"value": np.ones((2, 2))}])
    expect = VectorField.from_expression(grid, ["3 * r**2", "5 * r**2"])
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)
