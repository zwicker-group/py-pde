"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField, SphericalSymGrid, Tensor2Field, VectorField

pytest.importorskip("jax")


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_findiff_sph(backend):
    """Test operator for a simple spherical grid."""
    grid = SphericalSymGrid(1.5, 3)
    _, r1, r2 = grid.axes_coords[0]
    assert grid.discretization == (0.5,)
    s = ScalarField(grid, [1, 2, 4])
    v = VectorField(grid, [[1, 2, 4], [0] * 3, [0] * 3])

    # test gradient
    grad = s.gradient(bc={"r-": "derivative", "r+": "value"}, backend=backend)
    np.testing.assert_allclose(grad.data[0, :], [1, 3, -6])
    grad = s.gradient(bc="derivative", backend=backend)
    np.testing.assert_allclose(grad.data[0, :], [1, 3, 2])
    grad = s.gradient(bc="derivative", method="forward", backend=backend)
    np.testing.assert_allclose(grad.data[0, :], [2, 4, 0])
    grad = s.gradient(bc="derivative", method="backward", backend=backend)
    np.testing.assert_allclose(grad.data[0, :], [0, 2, 4])

    # test divergence
    div = v.divergence(
        bc={"r-": "derivative", "r+": "value"}, conservative=False, backend=backend
    )
    np.testing.assert_allclose(div.data, [9, 3 + 4 / r1, -6 + 8 / r2], rtol=1e-6)
    div = v.divergence(
        bc="derivative", method="forward", conservative=False, backend=backend
    )
    np.testing.assert_allclose(div.data, [10, 4 + 4 / r1, 8 / r2], rtol=1e-6)
    div = v.divergence(
        bc="derivative", method="backward", conservative=False, backend=backend
    )
    np.testing.assert_allclose(div.data, [8, 2 + 4 / r1, 4 + 8 / r2], rtol=1e-6)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_conservative_sph(backend):
    """Test whether the integral over a divergence vanishes."""
    grid = SphericalSymGrid((0, 2), 50)
    expr = "1 / cosh((r - 1) * 10)"

    # test divergence of vector field
    for method in ["central", "forward", "backward"]:
        vf = VectorField.from_expression(grid, [expr, 0, 0])
        div = vf.divergence(
            bc="derivative", conservative=True, method=method, backend=backend
        )
        assert div.integral == pytest.approx(0, abs=1e-2)

    # test laplacian of scalar field
    lap = vf[0].laplace("derivative", backend=backend)
    # use bigger tolerance in case of float32 backend
    assert lap.integral == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
@pytest.mark.parametrize(
    ("op_name", "field"),
    [
        ("laplace", ScalarField),
        ("divergence", VectorField),
        ("gradient", ScalarField),
    ],
)
def test_small_annulus_sph(backend, op_name, field, rng):
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
        field(g, data=f.data).apply_operator(
            op_name, "auto_periodic_neumann", backend=backend
        )
        for g in grids
    ]

    np.testing.assert_almost_equal(res[0].data, res[1].data, decimal=5)
    assert np.linalg.norm(res[0].data - res[2].data) > 1e-3


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_grid_laplace_sph(backend):
    """Test the spherical implementation of the laplace operator."""
    grid_sph = SphericalSymGrid(9, 11)
    grid_cart = CartesianGrid([[-5, 5], [-5, 5], [-5, 5]], [12, 10, 11])

    a_1d = ScalarField.from_expression(grid_sph, "cos(r)")
    a_3d = a_1d.interpolate_to_grid(grid_cart)

    b_3d = a_3d.laplace("auto_periodic_neumann", backend=backend)
    b_1d = a_1d.laplace("auto_periodic_neumann", backend=backend)
    b_1d_3 = b_1d.interpolate_to_grid(grid_cart)

    i = slice(1, -1)  # do not compare boundary points
    np.testing.assert_allclose(
        b_1d_3.data[i, i, i], b_3d.data[i, i, i], rtol=0.2, atol=0.2
    )


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
@pytest.mark.parametrize("r_inner", [0, 1])
def test_gradient_squared_sph(backend, r_inner, rng):
    """Compare gradient squared operator for spherical grids."""
    grid = SphericalSymGrid((r_inner, 5), 64)
    field = ScalarField.random_harmonic(grid, modes=1, rng=rng)
    s1 = field.gradient("auto_periodic_neumann", backend=backend).to_scalar(
        "squared_sum"
    )
    s2 = field.gradient_squared("auto_periodic_neumann", central=True, backend=backend)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.1, atol=0.1)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False, backend=backend)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.1, atol=0.1)
    assert not np.array_equal(s2.data, s3.data)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_grid_div_grad_sph(backend):
    """Compare div grad to laplacian for spherical grids."""
    grid = SphericalSymGrid(2 * np.pi, 16)
    field = ScalarField.from_expression(grid, "cos(r)")

    a = field.laplace("derivative", backend=backend)
    b = field.gradient("derivative", backend=backend).divergence(
        "value", backend=backend
    )
    res = ScalarField.from_expression(grid, "-2 * sin(r) / r - cos(r)")

    # do not test the radial boundary points
    np.testing.assert_allclose(a.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)
    np.testing.assert_allclose(b.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.1)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_examples_scalar_sph(backend):
    """Compare derivatives of scalar fields for spherical grids."""
    grid = SphericalSymGrid(1, 32)
    sf = ScalarField.from_expression(grid, "r**3")

    # gradient
    res = sf.gradient(
        {"r-": {"derivative": 0}, "r+": {"derivative": 3}}, backend=backend
    )
    expect = VectorField.from_expression(grid, ["3 * r**2", 0, 0])
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # gradient squared
    expect = ScalarField.from_expression(grid, "9 * r**4")
    for c in [True, False]:
        res = sf.gradient_squared(
            {"r-": {"derivative": 0}, "r+": {"value": 1}}, central=c, backend=backend
        )
        np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # laplace
    res = sf.laplace(
        {"r-": {"derivative": 0}, "r+": {"derivative": 3}}, backend=backend
    )
    expect = ScalarField.from_expression(grid, "12 * r")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("backend", ["jax"], indirect=True)
def test_tensor_double_divergence_sph(backend):
    """Test double divergence of a tensor field against analytical expression."""
    grid = SphericalSymGrid([0.5, 1], 12)
    tf = Tensor2Field.from_expression(
        grid, [["r**4", 0, 0], [0, "r**3", 0], [0, 0, "r**3"]]
    )
    res = tf.apply_operator("tensor_double_divergence", bc="curvature", backend=backend)
    expect = ScalarField.from_expression(grid, "2 * r * (15 * r - 4)")
    np.testing.assert_allclose(res.data[1:-1], expect.data[1:-1], rtol=0.01)
