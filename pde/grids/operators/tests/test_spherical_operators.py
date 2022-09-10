"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import (
    CartesianGrid,
    ScalarField,
    SphericalSymGrid,
    Tensor2Field,
    VectorField,
    solve_poisson_equation,
)


def test_findiff_sph():
    """test operator for a simple spherical grid"""
    grid = SphericalSymGrid(1.5, 3)
    _, r1, r2 = grid.axes_coords[0]
    assert grid.discretization == (0.5,)
    s = ScalarField(grid, [1, 2, 4])
    v = VectorField(grid, [[1, 2, 4], [0] * 3, [0] * 3])

    # test gradient
    grad = s.gradient(bc=["derivative", "value"])
    np.testing.assert_allclose(grad.data[0, :], [1, 3, -6])
    grad = s.gradient(bc="derivative")
    np.testing.assert_allclose(grad.data[0, :], [1, 3, 2])

    # test divergence
    div = v.divergence(bc=["derivative", "value"], conservative=False)
    np.testing.assert_allclose(div.data, [9, 3 + 4 / r1, -6 + 8 / r2])
    div = v.divergence(bc="derivative", conservative=False)
    np.testing.assert_allclose(div.data, [9, 3 + 4 / r1, 2 + 8 / r2])


def test_conservative_sph():
    """test whether the integral over a divergence vanishes"""
    grid = SphericalSymGrid((0, 2), 50)
    expr = "1 / cosh((r - 1) * 10)"

    # test divergence of vector field
    vf = VectorField.from_expression(grid, [expr, 0, 0])
    div = vf.divergence(bc="derivative", conservative=True)
    assert div.integral == pytest.approx(0, abs=1e-2)

    # test laplacian of scalar field
    lap = vf[0].laplace("derivative")
    assert lap.integral == pytest.approx(0, abs=1e-13)

    # test double divergence of tensor field
    expressions = [[expr, 0, 0], [0, expr, 0], [0, 0, expr]]
    tf = Tensor2Field.from_expression(grid, expressions)
    res = tf._apply_operator(
        "tensor_double_divergence", bc="derivative", conservative=True
    )
    assert res.integral == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize(
    "op_name,field",
    [
        ("laplace", ScalarField),
        ("divergence", VectorField),
        ("gradient", ScalarField),
    ],
)
def test_small_annulus_sph(op_name, field):
    """test whether a small annulus gives the same result as a sphere"""
    grids = [
        SphericalSymGrid((0, 1), 8),
        SphericalSymGrid((1e-8, 1), 8),
        SphericalSymGrid((0.1, 1), 8),
    ]

    f = field.random_uniform(grids[0])
    if field is VectorField:
        f.data[1] = 0

    res = [
        field(g, data=f.data)._apply_operator(op_name, "auto_periodic_neumann")
        for g in grids
    ]

    np.testing.assert_almost_equal(res[0].data, res[1].data, decimal=5)
    assert np.linalg.norm(res[0].data - res[2].data) > 1e-3


def test_grid_laplace():
    """test the polar implementation of the laplace operator"""
    grid_sph = SphericalSymGrid(9, 11)
    grid_cart = CartesianGrid([[-5, 5], [-5, 5], [-5, 5]], [12, 10, 11])

    a_1d = ScalarField.from_expression(grid_sph, "cos(r)")
    a_3d = a_1d.interpolate_to_grid(grid_cart)

    b_3d = a_3d.laplace("auto_periodic_neumann")
    b_1d = a_1d.laplace("auto_periodic_neumann")
    b_1d_3 = b_1d.interpolate_to_grid(grid_cart)

    i = slice(1, -1)  # do not compare boundary points
    np.testing.assert_allclose(
        b_1d_3.data[i, i, i], b_3d.data[i, i, i], rtol=0.2, atol=0.2
    )


@pytest.mark.parametrize("r_inner", (0, 1))
def test_gradient_squared(r_inner):
    """compare gradient squared operator"""
    grid = SphericalSymGrid((r_inner, 5), 64)
    field = ScalarField.random_harmonic(grid, modes=1)
    s1 = field.gradient("auto_periodic_neumann").to_scalar("squared_sum")
    s2 = field.gradient_squared("auto_periodic_neumann", central=True)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.1, atol=0.1)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.1, atol=0.1)
    assert not np.array_equal(s2.data, s3.data)


def test_grid_div_grad_sph():
    """compare div grad to laplacian"""
    grid = SphericalSymGrid(2 * np.pi, 16)
    field = ScalarField.from_expression(grid, "cos(r)")

    a = field.laplace("derivative")
    b = field.gradient("derivative").divergence("value")
    res = ScalarField.from_expression(grid, "-2 * sin(r) / r - cos(r)")

    # do not test the radial boundary points
    np.testing.assert_allclose(a.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)
    np.testing.assert_allclose(b.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)


def test_poisson_solver_spherical():
    """test the poisson solver on Polar grids"""
    for grid in [SphericalSymGrid(4, 8), SphericalSymGrid([2, 4], 8)]:
        for bc_val in ["auto_periodic_neumann", {"value": 1}]:
            bcs = grid.get_boundary_conditions(bc_val)
            d = ScalarField.random_uniform(grid)
            d -= d.average  # balance the right hand side
            sol = solve_poisson_equation(d, bcs)
            test = sol.laplace(bcs)
            np.testing.assert_allclose(
                test.data, d.data, err_msg=f"bcs={bc_val}, grid={grid}", rtol=1e-6
            )


def test_examples_scalar_sph():
    """compare derivatives of scalar fields for spherical grids"""
    grid = SphericalSymGrid(1, 32)
    sf = ScalarField.from_expression(grid, "r**3")

    # gradient
    res = sf.gradient([{"derivative": 0}, {"derivative": 3}])
    expect = VectorField.from_expression(grid, ["3 * r**2", 0, 0])
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # gradient squared
    expect = ScalarField.from_expression(grid, "9 * r**4")
    for c in [True, False]:
        res = sf.gradient_squared([{"derivative": 0}, {"value": 1}], central=c)
        np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # laplace
    res = sf.laplace([{"derivative": 0}, {"derivative": 3}])
    expect = ScalarField.from_expression(grid, "12 * r")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_vector_sph():
    """compare derivatives of vector fields for spherical grids"""
    grid = SphericalSymGrid(1, 32)

    # divergence
    vf = VectorField.from_expression(grid, ["r**3", 0, "r**2"])
    res = vf.divergence([{"derivative": 0}, {"value": 1}])
    expect = ScalarField.from_expression(grid, "5 * r**2")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # vector gradient
    vf = VectorField.from_expression(grid, ["r**3", 0, 0])
    res = vf.gradient([{"derivative": 0}, {"value": [1, 1, 1]}])
    expr = [["3 * r**2", 0, 0], [0, "r**2", 0], [0, 0, "r**2"]]
    expect = Tensor2Field.from_expression(grid, expr)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("conservative", [True, False])
def test_examples_tensor_sph(conservative):
    """compare derivatives of tensorial fields for spherical grids"""
    # test explicit expression for which we know the results
    grid = SphericalSymGrid(1, 32)
    expressions = [["r**4", 0, 0], [0, "r**3", 0], [0, 0, "r**3"]]
    tf = Tensor2Field.from_expression(grid, expressions)

    # tensor divergence
    bc = [{"derivative": 0}, {"normal_derivative": [4, 3, 3]}]
    res = tf.divergence(bc, conservative=conservative)
    expect = VectorField.from_expression(grid, ["2 * r**2 * (3 * r - 1)", 0, 0])
    if conservative:
        np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)
    else:
        np.testing.assert_allclose(
            res.data[:, 1:-1], expect.data[:, 1:-1], rtol=0.1, atol=0.1
        )


def test_tensor_sph_symmetry():
    """test treatment of symmetric tensor field"""
    grid = SphericalSymGrid(1, 16)
    vf = VectorField.from_expression(grid, ["r**2", 0, 0])
    vf_grad = vf.gradient(["derivative", {"derivative": 2}])
    strain = vf_grad + vf_grad.transpose()

    bcs = [{"value": 0}, {"normal_derivative": [4, 0, 0]}]
    strain_div = strain.divergence(bcs, safe=False)
    np.testing.assert_allclose(strain_div.data[0], 8)
    np.testing.assert_allclose(strain_div.data[1:], 0)


def test_tensor_div_div_analytical():
    """test double divergence of a tensor field against analytical expression"""
    grid = SphericalSymGrid([0.5, 1], 12)
    tf = Tensor2Field.from_expression(
        grid, [["r**4", 0, 0], [0, "r**3", 0], [0, 0, "r**3"]]
    )
    res = tf._apply_operator("tensor_double_divergence", bc="curvature")
    expect = ScalarField.from_expression(grid, "2 * r * (15 * r - 4)")
    np.testing.assert_allclose(res.data[1:-1], expect.data[1:-1], rtol=0.01)


@pytest.mark.parametrize("conservative", [True, False])
def test_tensor_div_div(conservative):
    """test double divergence of a tensor field by comparison with two divergences"""
    grid = SphericalSymGrid([0, 1], 64)
    expr = "r * tanh((0.5 - r) * 10)"
    bc = "auto_periodic_neumann"

    # test radial part
    tf = Tensor2Field.from_expression(grid, [[expr, 0, 0], [0, 0, 0], [0, 0, 0]])
    res = tf._apply_operator(
        "tensor_double_divergence", bc=bc, conservative=conservative
    )
    est = tf.divergence(bc).divergence(bc)
    np.testing.assert_allclose(res.data[2:-2], est.data[2:-2], rtol=0.02, atol=1)

    # test angular part
    tf = Tensor2Field.from_expression(grid, [[0, 0, 0], [0, expr, 0], [0, 0, expr]])
    res = tf._apply_operator(
        "tensor_double_divergence", bc=bc, conservative=conservative
    )
    est = tf.divergence(bc).divergence(bc)
    np.testing.assert_allclose(res.data[2:-2], est.data[2:-2], rtol=0.02, atol=1)
