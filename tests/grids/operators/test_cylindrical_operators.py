"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import (
    CartesianGrid,
    CylindricalSymGrid,
    ScalarField,
    Tensor2Field,
    VectorField,
    solve_poisson_equation,
)
from pde.grids.operators.common import make_laplace_from_matrix
from pde.grids.operators.cylindrical_sym import _get_laplace_matrix


def test_laplacian_field_cyl():
    """Test the gradient operator."""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], [8, 16], periodic_z=True)
    r, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    s = ScalarField(grid, data=np.cos(r) + np.sin(z))
    s_lap = s.laplace(bc="auto_periodic_neumann")
    assert s_lap.data.shape == (8, 16)
    res = -np.cos(r) - np.sin(r) / r - np.sin(z)
    np.testing.assert_allclose(s_lap.data, res, rtol=0.1, atol=0.1)


def test_gradient_field_cyl():
    """Test the gradient operator."""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], [8, 16], periodic_z=True)
    r, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    s = ScalarField(grid, data=np.cos(r) + np.sin(z))
    v = s.gradient(bc="auto_periodic_neumann")
    assert v.data.shape == (3, 8, 16)
    np.testing.assert_allclose(v.data[0], -np.sin(r), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(v.data[1], np.cos(z), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(v.data[2], 0, rtol=0.1, atol=0.1)


def test_divergence_field_cyl():
    """Test the divergence operator."""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], [16, 32], periodic_z=True)
    v = VectorField.from_expression(grid, ["cos(r) + sin(z)**2", "z * cos(r)**2", 0])
    s = v.divergence(bc="auto_periodic_neumann")
    assert s.data.shape == grid.shape
    res = ScalarField.from_expression(
        grid, "cos(r)**2 - sin(r) + (cos(r) + sin(z)**2) / r"
    )
    np.testing.assert_allclose(
        s.data[1:-1, 1:-1], res.data[1:-1, 1:-1], rtol=0.1, atol=0.1
    )

    # test with inner hole in cylindrical grid
    grid2 = CylindricalSymGrid(
        (np.pi, 2 * np.pi), [0, 2 * np.pi], [8, 32], periodic_z=True
    )
    v2 = VectorField.from_expression(grid2, ["cos(r) + sin(z)**2", "z * cos(r)**2", 0])
    s2 = v2.divergence(bc="auto_periodic_neumann")
    assert s2.data.shape == grid2.shape
    res2 = ScalarField.from_expression(
        grid2, "cos(r)**2 - sin(r) + (cos(r) + sin(z)**2) / r"
    )
    np.testing.assert_allclose(res2.data[1:-1, 1:-1], res.data[9:-1, 1:-1])
    np.testing.assert_allclose(
        s2.data[1:-1, 1:-1], res2.data[1:-1, 1:-1], rtol=0.1, atol=0.1
    )


def test_vector_gradient_divergence_field_cyl():
    """Test the divergence operator."""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], [8, 16], periodic_z=True)
    r, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(r) + np.sin(z) ** 2, np.cos(r) ** 2 + np.sin(z), np.zeros_like(r)]
    v = VectorField(grid, data=data)
    t = v.gradient(bc="auto_periodic_neumann")
    assert t.data.shape == (3, 3, 8, 16)
    v = t.divergence(bc="auto_periodic_neumann")
    assert v.data.shape == (3, 8, 16)


def test_findiff_cyl():
    """Test operator for a simple cylindrical grid.

    Note that we only really test the polar symmetry
    """
    grid = CylindricalSymGrid(1.5, [0, 1], (3, 2), periodic_z=True)
    _, r1, r2 = grid.axes_coords[0]
    np.testing.assert_array_equal(grid.discretization, np.full(2, 0.5))
    s = ScalarField(grid, [[1, 1], [2, 2], [4, 4]])

    # test laplace
    lap = s.laplace(bc={"r": {"value": 3}, "z": "periodic"})
    y1 = 4 + 3 / r1
    y2 = -16
    np.testing.assert_allclose(lap.data, [[8, 8], [y1, y1], [y2, y2]])
    lap = s.laplace(bc={"r": {"derivative": 3}, "z": "periodic"})
    y2 = -2 + 3.5 / r2
    np.testing.assert_allclose(lap.data, [[8, 8], [y1, y1], [y2, y2]])


def test_grid_laplace():
    """Test the cylindrical implementation of the laplace operator."""
    grid_cyl = CylindricalSymGrid(7, (0, 4), (4, 4))
    grid_cart = CartesianGrid([[-5, 5], [-5, 5], [0, 4]], [10, 10, 4])

    a_2d = ScalarField.from_expression(grid_cyl, expression="exp(-5 * r) * cos(z / 3)")
    a_3d = a_2d.interpolate_to_grid(grid_cart)

    b_2d = a_2d.laplace("auto_periodic_neumann")
    b_3d = a_3d.laplace("auto_periodic_neumann")
    b_2d_3 = b_2d.interpolate_to_grid(grid_cart)

    np.testing.assert_allclose(b_2d_3.data, b_3d.data, rtol=0.2, atol=0.2)


def test_gradient_squared_cyl(rng):
    """Compare gradient squared operator."""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], 64)
    field = ScalarField.random_harmonic(grid, modes=1, rng=rng)
    s1 = field.gradient("auto_periodic_neumann").to_scalar("squared_sum")
    s2 = field.gradient_squared("auto_periodic_neumann", central=True)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.2, atol=0.2)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.2, atol=0.2)
    assert not np.array_equal(s2.data, s3.data)


def test_grid_div_grad_cyl():
    """Compare div grad to laplacian."""
    grid = CylindricalSymGrid(2 * np.pi, (0, 2 * np.pi), (16, 16), periodic_z=True)
    field = ScalarField.from_expression(grid, "cos(r) + sin(z)")

    bcs = grid.get_boundary_conditions()
    a = field.laplace(bcs)
    c = field.gradient(bcs)
    b = c.divergence("auto_periodic_curvature")
    res = ScalarField.from_expression(grid, "-sin(r)/r - cos(r) - sin(z)")

    # do not test the radial boundary points
    np.testing.assert_allclose(a.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.05)
    np.testing.assert_allclose(b.data[1:-1], res.data[1:-1], rtol=0.1, atol=0.05)


def test_examples_scalar_cyl():
    """Compare derivatives of scalar fields for cylindrical grids."""
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32)
    expr = "r**3 * sin(z)"
    sf = ScalarField.from_expression(grid, expr)
    bcs = {
        "r-": {"derivative": 0},
        "r+": {"value": expr},
        "z-": {"value": expr},
        "z+": {"value": expr},
    }

    # gradient - The coordinates are ordered as (r, z, φ) in py-pde
    res = sf.gradient(bcs)
    expect = VectorField.from_expression(
        grid, ["3 * r**2 * sin(z)", "r**3 * cos(z)", 0]
    )
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # gradient squared
    expect = ScalarField.from_expression(
        grid, "r**6 * cos(z)**2 + 9 * r**4 * sin(z)**2"
    )
    res = sf.gradient_squared(bcs, central=True)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # laplace
    bcs["r+"] = {"curvature": "6 * sin(z)"}  # adjust BC to fit laplacian better
    res = sf.laplace(bcs)
    expect = ScalarField.from_expression(grid, "9 * r * sin(z) - r**3 * sin(z)")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_vector_cyl():
    """Compare derivatives of vector fields for cylindrical grids."""
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32)
    e_r = "r**3 * sin(z)"
    e_φ = "r**2 * sin(z)"
    e_z = "r**4 * cos(z)"
    vf = VectorField.from_expression(grid, [e_r, e_z, e_φ])
    bcs = {
        "r-": {"normal_derivative": 0},
        "r+": {"normal_value": "r**3 * sin(z)"},
        "z": {"normal_curvature": "-r**4 * cos(z)"},
    }

    # divergence
    res = vf.divergence(bcs)
    expect = ScalarField.from_expression(grid, "4 * r**2 * sin(z) - r**4 * sin(z)")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # vector Laplacian
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32, periodic_z=True)
    vf = VectorField.from_expression(grid, ["r**3 * sin(z)"] * 3)
    val_r_outer = np.broadcast_to(6 * np.sin(grid.axes_coords[1]), (3, 32))
    bcs = {"r-": {"derivative": 0}, "r+": {"curvature": val_r_outer}, "z": "periodic"}
    res = vf.laplace(bcs)
    expr = [
        "8 * r * sin(z) - r**3 * sin(z)",
        "9 * r * sin(z) - r**3 * sin(z)",
        "8 * r * sin(z) - r**3 * sin(z)",
    ]
    expect = VectorField.from_expression(grid, expr)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # vector gradient
    bcs = {"r-": {"derivative": 0}, "r+": {"curvature": val_r_outer}, "z": "periodic"}
    res = vf.gradient(bcs)
    expr = [
        ["3 * r**2 * sin(z)", "r**3 * cos(z)", "-r**2 * sin(z)"],
        ["3 * r**2 * sin(z)", "r**3 * cos(z)", 0],
        ["3 * r**2 * sin(z)", "r**3 * cos(z)", "r**2 * sin(z)"],
    ]
    expect = Tensor2Field.from_expression(grid, expr)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_tensor_cyl():
    """Compare derivatives of tensorial fields for cylindrical grids."""
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32, periodic_z=True)
    tf = Tensor2Field.from_expression(grid, [["r**3 * sin(z)"] * 3] * 3)

    # tensor divergence
    rs, zs = grid.axes_coords
    val_r_outer = np.broadcast_to(6 * rs * np.sin(zs), (3, 32))
    bcs = {
        "r-": {"normal_derivative": 0},
        "r+": {"normal_curvature": val_r_outer},
        "z": "periodic",
    }
    res = tf.divergence(bcs)
    expect = VectorField.from_expression(
        grid,
        [
            "r**2 * (r * cos(z) + 3 * sin(z))",
            "r**2 * (r * cos(z) + 4 * sin(z))",
            "r**2 * (r * cos(z) + 5 * sin(z))",
        ],
    )
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("r_inner", [0, 1])
def test_laplace_matrix(r_inner, rng):
    """Test laplace operator implemented using matrix multiplication."""
    grid = CylindricalSymGrid((r_inner, 2), (2.5, 4.3), 16)
    if r_inner == 0:
        bcs = {"r": "neumann"}
    else:
        bcs = {"r": {"value": "sin(r)"}}
    bcs["z"] = {"derivative": "cos(r) + z"}
    bcs = grid.get_boundary_conditions(bcs)
    laplace = make_laplace_from_matrix(*_get_laplace_matrix(bcs))

    field = ScalarField.random_uniform(grid, rng=rng)
    res1 = field.laplace(bcs)
    res2 = laplace(field.data)

    np.testing.assert_allclose(res1.data, res2)


@pytest.mark.parametrize("r_inner", [0, 1])
def test_poisson_solver_cylindrical(r_inner, rng):
    """Test the poisson solver on Cylindrical grids."""
    grid = CylindricalSymGrid((r_inner, 2), (2.5, 4.3), 16)
    if r_inner == 0:
        bcs = {"r": "neumann", "z": {"value": "cos(r) + z"}}
    else:
        bcs = {"r": {"value": "sin(r)"}, "z": {"derivative": "cos(r) + z"}}
    d = ScalarField.random_uniform(grid, rng=rng)
    d -= d.average  # balance the right hand side
    sol = solve_poisson_equation(d, bcs)
    test = sol.laplace(bcs)
    np.testing.assert_allclose(
        test.data, d.data, err_msg=f"bcs={bcs}, grid={grid}", rtol=1e-6
    )
