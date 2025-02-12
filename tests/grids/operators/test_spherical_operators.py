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
from pde.grids.operators.common import make_laplace_from_matrix
from pde.grids.operators.spherical_sym import _get_laplace_matrix


def test_findiff_sph():
    """Test operator for a simple spherical grid."""
    grid = SphericalSymGrid(1.5, 3)
    _, r1, r2 = grid.axes_coords[0]
    assert grid.discretization == (0.5,)
    s = ScalarField(grid, [1, 2, 4])
    v = VectorField(grid, [[1, 2, 4], [0] * 3, [0] * 3])

    # test gradient
    grad = s.gradient(bc={"r-": "derivative", "r+": "value"})
    np.testing.assert_allclose(grad.data[0, :], [1, 3, -6])
    grad = s.gradient(bc="derivative")
    np.testing.assert_allclose(grad.data[0, :], [1, 3, 2])
    grad = s.gradient(bc="derivative", method="forward")
    np.testing.assert_allclose(grad.data[0, :], [2, 4, 0])
    grad = s.gradient(bc="derivative", method="backward")
    np.testing.assert_allclose(grad.data[0, :], [0, 2, 4])

    # test divergence
    div = v.divergence(bc={"r-": "derivative", "r+": "value"}, conservative=False)
    np.testing.assert_allclose(div.data, [9, 3 + 4 / r1, -6 + 8 / r2])
    div = v.divergence(bc="derivative", method="forward", conservative=False)
    np.testing.assert_allclose(div.data, [10, 4 + 4 / r1, 8 / r2])
    div = v.divergence(bc="derivative", method="backward", conservative=False)
    np.testing.assert_allclose(div.data, [8, 2 + 4 / r1, 4 + 8 / r2])


def test_conservative_sph():
    """Test whether the integral over a divergence vanishes."""
    grid = SphericalSymGrid((0, 2), 50)
    expr = "1 / cosh((r - 1) * 10)"

    # test divergence of vector field
    for method in ["central", "forward", "backward"]:
        vf = VectorField.from_expression(grid, [expr, 0, 0])
        div = vf.divergence(bc="derivative", conservative=True, method=method)
        assert div.integral == pytest.approx(0, abs=1e-2)

    # test laplacian of scalar field
    lap = vf[0].laplace("derivative")
    assert lap.integral == pytest.approx(0, abs=1e-13)

    # test double divergence of tensor field
    expressions = [[expr, 0, 0], [0, expr, 0], [0, 0, expr]]
    tf = Tensor2Field.from_expression(grid, expressions)
    res = tf.apply_operator(
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
def test_small_annulus_sph(op_name, field, rng):
    """Test whether a small annulus gives the same result as a sphere."""
    grids = [
        SphericalSymGrid((0, 1), 8),
        SphericalSymGrid((1e-8, 1), 8),
        SphericalSymGrid((0.1, 1), 8),
    ]

    f = field.random_uniform(grids[0], rng=rng)
    if field is VectorField:
        f.data[1] = 0

    res = [
        field(g, data=f.data).apply_operator(op_name, "auto_periodic_neumann")
        for g in grids
    ]

    np.testing.assert_almost_equal(res[0].data, res[1].data, decimal=5)
    assert np.linalg.norm(res[0].data - res[2].data) > 1e-3


def test_grid_laplace():
    """Test the polar implementation of the laplace operator."""
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


@pytest.mark.parametrize("r_inner", [0, 1])
def test_gradient_squared(r_inner, rng):
    """Compare gradient squared operator."""
    grid = SphericalSymGrid((r_inner, 5), 64)
    field = ScalarField.random_harmonic(grid, modes=1, rng=rng)
    s1 = field.gradient("auto_periodic_neumann").to_scalar("squared_sum")
    s2 = field.gradient_squared("auto_periodic_neumann", central=True)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.1, atol=0.1)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.1, atol=0.1)
    assert not np.array_equal(s2.data, s3.data)


def test_grid_div_grad_sph():
    """Compare div grad to laplacian."""
    grid = SphericalSymGrid(2 * np.pi, 16)
    field = ScalarField.from_expression(grid, "cos(r)")

    a = field.laplace("derivative")
    b = field.gradient("derivative").divergence("value")
    res = ScalarField.from_expression(grid, "-2 * sin(r) / r - cos(r)")

    # do not test the radial boundary points
    np.testing.assert_allclose(a.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)
    np.testing.assert_allclose(b.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)


@pytest.mark.parametrize("grid", [SphericalSymGrid(4, 8), SphericalSymGrid([2, 4], 8)])
@pytest.mark.parametrize("bc_val", ["auto_periodic_neumann", {"value": 1}])
def test_poisson_solver_spherical(grid, bc_val, rng):
    """Test the poisson solver on Spherical grids."""
    bcs = grid.get_boundary_conditions(bc_val)
    d = ScalarField.random_uniform(grid, rng=rng)
    d -= d.average  # balance the right hand side
    sol = solve_poisson_equation(d, bcs)
    test = sol.laplace(bcs)
    np.testing.assert_allclose(
        test.data, d.data, err_msg=f"bcs={bc_val}, grid={grid}", rtol=1e-6
    )


def test_examples_scalar_sph():
    """Compare derivatives of scalar fields for spherical grids."""
    grid = SphericalSymGrid(1, 32)
    sf = ScalarField.from_expression(grid, "r**3")

    # gradient
    res = sf.gradient({"r-": {"derivative": 0}, "r+": {"derivative": 3}})
    expect = VectorField.from_expression(grid, ["3 * r**2", 0, 0])
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # gradient squared
    expect = ScalarField.from_expression(grid, "9 * r**4")
    for c in [True, False]:
        res = sf.gradient_squared(
            {"r-": {"derivative": 0}, "r+": {"value": 1}}, central=c
        )
        np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # laplace
    res = sf.laplace({"r-": {"derivative": 0}, "r+": {"derivative": 3}})
    expect = ScalarField.from_expression(grid, "12 * r")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_vector_sph_div():
    """Compare derivatives of vector fields for spherical grids."""
    grid = SphericalSymGrid(1, 32)
    vf = VectorField.from_expression(grid, ["r**3", 0, "r**2"])
    res = vf.divergence({"r-": {"derivative": 0}, "r+": {"value": 1}})
    expect = ScalarField.from_expression(grid, "5 * r**2")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_examples_vector_sph_grad(method):
    """Compare derivatives of vector fields for spherical grids."""
    grid = SphericalSymGrid(1, 32)
    vf = VectorField.from_expression(grid, ["r**3", 0, 0])
    res = vf.gradient(
        {"r-": {"derivative": 0}, "r+": {"value": [1, 1, 1]}}, method=method
    )
    expr = [["3 * r**2", 0, 0], [0, "r**2", 0], [0, 0, "r**2"]]
    expect = Tensor2Field.from_expression(grid, expr)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("conservative", [True, False])
def test_examples_tensor_sph(conservative):
    """Compare derivatives of tensorial fields for spherical grids."""
    # test explicit expression for which we know the results
    grid = SphericalSymGrid(1, 32)
    expressions = [["r**4", 0, 0], [0, "r**3", 0], [0, 0, "r**3"]]
    tf = Tensor2Field.from_expression(grid, expressions)

    # tensor divergence
    bc = {"r-": {"derivative": 0}, "r+": {"normal_derivative": [4, 3, 3]}}
    res = tf.divergence(bc, conservative=conservative)
    expect = VectorField.from_expression(grid, ["2 * r**2 * (3 * r - 1)", 0, 0])
    if conservative:
        np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)
    else:
        np.testing.assert_allclose(
            res.data[:, 1:-1], expect.data[:, 1:-1], rtol=0.1, atol=0.1
        )

    # test an edge case
    grid = SphericalSymGrid([0, 10], 50)
    tensor = Tensor2Field(grid)
    tensor[0, 0] = ScalarField.from_expression(grid, "tanh(r - 5)")
    tensor[1, 1] = tensor[0, 0]
    tensor[2, 2] = tensor[0, 0]

    bc = {
        "r-": {"value": [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]},
        "r+": {"derivative": 0},
    }
    div = tensor.divergence(bc=bc, conservative=conservative)

    expected = ScalarField.from_expression(grid, "cosh(5 - r)**-2")
    np.testing.assert_allclose(div[0].data, expected.data, atol=0.1)
    np.testing.assert_allclose(div[1].data, 0, atol=0.1)
    np.testing.assert_allclose(div[2].data, 0, atol=0.1)


def test_tensor_sph_symmetry():
    """Test treatment of symmetric tensor field."""
    grid = SphericalSymGrid(1, 16)
    vf = VectorField.from_expression(grid, ["r**2", 0, 0])
    vf_grad = vf.gradient({"r-": "derivative", "r+": {"derivative": 2}})
    strain = vf_grad + vf_grad.transpose()

    expect = ScalarField.from_expression(grid, "2*r").data
    np.testing.assert_allclose(strain.data[0, 0], 2 * expect)
    np.testing.assert_allclose(strain.data[1, 1], expect)
    np.testing.assert_allclose(strain.data[2, 2], expect)

    bcs = {"r-": {"value": 0}, "r+": {"normal_derivative": [4, 0, 0]}}
    strain_div = strain.divergence(bcs)
    np.testing.assert_allclose(strain_div.data[0], 8)
    np.testing.assert_allclose(strain_div.data[1:], 0)


def test_tensor_div_div_analytical():
    """Test double divergence of a tensor field against analytical expression."""
    grid = SphericalSymGrid([0.5, 1], 12)
    tf = Tensor2Field.from_expression(
        grid, [["r**4", 0, 0], [0, "r**3", 0], [0, 0, "r**3"]]
    )
    res = tf.apply_operator("tensor_double_divergence", bc="curvature")
    expect = ScalarField.from_expression(grid, "2 * r * (15 * r - 4)")
    np.testing.assert_allclose(res.data[1:-1], expect.data[1:-1], rtol=0.01)


@pytest.mark.parametrize("conservative", [True, False])
def test_tensor_div_div(conservative):
    """Test double divergence of a tensor field by comparison with two divergences."""
    grid = SphericalSymGrid([0, 1], 64)
    expr = "r * tanh((0.5 - r) * 10)"
    bc = "auto_periodic_neumann"

    # test radial part
    tf = Tensor2Field.from_expression(grid, [[expr, 0, 0], [0, 0, 0], [0, 0, 0]])
    res = tf.apply_operator(
        "tensor_double_divergence", bc=bc, conservative=conservative
    )
    est = tf.divergence(bc).divergence(bc)
    np.testing.assert_allclose(res.data[2:-2], est.data[2:-2], rtol=0.02, atol=1)

    # test angular part
    tf = Tensor2Field.from_expression(grid, [[0, 0, 0], [0, expr, 0], [0, 0, expr]])
    res = tf.apply_operator(
        "tensor_double_divergence", bc=bc, conservative=conservative
    )
    est = tf.divergence(bc).divergence(bc)
    np.testing.assert_allclose(res.data[2:-2], est.data[2:-2], rtol=0.02, atol=1)


@pytest.mark.parametrize("r_inner", [0, 1])
def test_laplace_matrix(r_inner, rng):
    """Test laplace operator implemented using matrix multiplication."""
    grid = SphericalSymGrid((r_inner, 2), 16)
    if r_inner == 0:
        bcs = grid.get_boundary_conditions("neumann")
    else:
        bcs = grid.get_boundary_conditions({"value": "sin(r)"})
    laplace = make_laplace_from_matrix(*_get_laplace_matrix(bcs))

    field = ScalarField.random_uniform(grid, rng=rng)
    res1 = field.laplace(bcs)
    res2 = laplace(field.data)

    np.testing.assert_allclose(res1.data, res2)
