"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde import (
    CartesianGrid,
    CylindricalSymGrid,
    ScalarField,
    Tensor2Field,
    VectorField,
)


def test_laplacian_field_cyl():
    """test the gradient operator"""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], [8, 16], periodic_z=True)
    r, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    s = ScalarField(grid, data=np.cos(r) + np.sin(z))
    s_lap = s.laplace(bc="auto_periodic_neumann")
    assert s_lap.data.shape == (8, 16)
    res = -np.cos(r) - np.sin(r) / r - np.sin(z)
    np.testing.assert_allclose(s_lap.data, res, rtol=0.1, atol=0.1)


def test_gradient_field_cyl():
    """test the gradient operator"""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], [8, 16], periodic_z=True)
    r, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    s = ScalarField(grid, data=np.cos(r) + np.sin(z))
    v = s.gradient(bc="auto_periodic_neumann")
    assert v.data.shape == (3, 8, 16)
    np.testing.assert_allclose(v.data[0], -np.sin(r), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(v.data[1], np.cos(z), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(v.data[2], 0, rtol=0.1, atol=0.1)


def test_divergence_field_cyl():
    """test the divergence operator"""
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


def test_vector_gradient_divergence_field_cyl():
    """test the divergence operator"""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], [8, 16], periodic_z=True)
    r, z = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(r) + np.sin(z) ** 2, np.cos(r) ** 2 + np.sin(z), np.zeros_like(r)]
    v = VectorField(grid, data=data)
    t = v.gradient(bc="auto_periodic_neumann")
    assert t.data.shape == (3, 3, 8, 16)
    v = t.divergence(bc="auto_periodic_neumann")
    assert v.data.shape == (3, 8, 16)


def test_findiff_cyl():
    """test operator for a simple cylindrical grid. Note that we only
    really test the polar symmetry"""
    grid = CylindricalSymGrid(1.5, [0, 1], (3, 2), periodic_z=True)
    _, r1, r2 = grid.axes_coords[0]
    np.testing.assert_array_equal(grid.discretization, np.full(2, 0.5))
    s = ScalarField(grid, [[1, 1], [2, 2], [4, 4]])

    # test laplace
    lap = s.laplace(bc=[{"type": "value", "value": 3}, "periodic"])
    y1 = 4 + 3 / r1
    y2 = -16
    np.testing.assert_allclose(lap.data, [[8, 8], [y1, y1], [y2, y2]])
    lap = s.laplace(bc=[{"type": "derivative", "value": 3}, "periodic"])
    y2 = -2 + 3.5 / r2
    np.testing.assert_allclose(lap.data, [[8, 8], [y1, y1], [y2, y2]])


def test_grid_laplace():
    """test the cylindrical implementation of the laplace operator"""
    grid_cyl = CylindricalSymGrid(7, (0, 4), (4, 4))
    grid_cart = CartesianGrid([[-5, 5], [-5, 5], [0, 4]], [10, 10, 4])

    a_2d = ScalarField.from_expression(grid_cyl, expression="exp(-5 * r) * cos(z / 3)")
    a_3d = a_2d.interpolate_to_grid(grid_cart)

    b_2d = a_2d.laplace("auto_periodic_neumann")
    b_3d = a_3d.laplace("auto_periodic_neumann")
    b_2d_3 = b_2d.interpolate_to_grid(grid_cart)

    np.testing.assert_allclose(b_2d_3.data, b_3d.data, rtol=0.2, atol=0.2)


def test_gradient_squared_cyl():
    """compare gradient squared operator"""
    grid = CylindricalSymGrid(2 * np.pi, [0, 2 * np.pi], 64)
    field = ScalarField.random_harmonic(grid, modes=1)
    s1 = field.gradient("auto_periodic_neumann").to_scalar("squared_sum")
    s2 = field.gradient_squared("auto_periodic_neumann", central=True)
    np.testing.assert_allclose(s1.data, s2.data, rtol=0.2, atol=0.2)
    s3 = field.gradient_squared("auto_periodic_neumann", central=False)
    np.testing.assert_allclose(s1.data, s3.data, rtol=0.2, atol=0.2)
    assert not np.array_equal(s2.data, s3.data)


def test_grid_div_grad_cyl():
    """compare div grad to laplacian"""
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
    """compare derivatives of scalar fields for cylindrical grids"""
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32)
    expr = "r**3 * sin(z)"
    sf = ScalarField.from_expression(grid, expr)
    bcs = [[{"derivative": 0}, {"value": expr}], [{"value": expr}, {"value": expr}]]

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
    bcs[0][1] = {"curvature": "6 * sin(z)"}  # adjust BC to fit laplacian better
    res = sf.laplace(bcs)
    expect = ScalarField.from_expression(grid, "9 * r * sin(z) - r**3 * sin(z)")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_vector_cyl():
    """compare derivatives of vector fields for cylindrical grids"""
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32)
    e_r = "r**3 * sin(z)"
    e_φ = "r**2 * sin(z)"
    e_z = "r**4 * cos(z)"
    vf = VectorField.from_expression(grid, [e_r, e_z, e_φ])
    bc_r = ({"normal_derivative": 0}, {"normal_value": "r**3 * sin(z)"})
    bc_z = {"normal_curvature": "-r**4 * cos(z)"}
    bcs = [bc_r, bc_z]

    # divergence
    res = vf.divergence(bcs)
    expect = ScalarField.from_expression(grid, "4 * r**2 * sin(z) - r**4 * sin(z)")
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # vector Laplacian
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32, periodic_z=True)
    vf = VectorField.from_expression(grid, ["r**3 * sin(z)"] * 3)
    val_r_outer = np.broadcast_to(6 * np.sin(grid.axes_coords[1]), (3, 32))
    bcs = [({"derivative": 0}, {"curvature": val_r_outer}), "periodic"]
    res = vf.laplace(bcs)
    expr = [
        "8 * r * sin(z) - r**3 * sin(z)",
        "9 * r * sin(z) - r**3 * sin(z)",
        "8 * r * sin(z) - r**3 * sin(z)",
    ]
    expect = VectorField.from_expression(grid, expr)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)

    # vector gradient
    bcs = [({"derivative": 0}, {"curvature": val_r_outer}), "periodic"]
    res = vf.gradient(bcs)
    expr = [
        ["3 * r**2 * sin(z)", "r**3 * cos(z)", "-r**2 * sin(z)"],
        ["3 * r**2 * sin(z)", "r**3 * cos(z)", 0],
        ["3 * r**2 * sin(z)", "r**3 * cos(z)", "r**2 * sin(z)"],
    ]
    expect = Tensor2Field.from_expression(grid, expr)
    np.testing.assert_allclose(res.data, expect.data, rtol=0.1, atol=0.1)


def test_examples_tensor_cyl():
    """compare derivatives of tensorial fields for cylindrical grids"""
    grid = CylindricalSymGrid(1, [0, 2 * np.pi], 32, periodic_z=True)
    tf = Tensor2Field.from_expression(grid, [["r**3 * sin(z)"] * 3] * 3)

    # tensor divergence
    rs, zs = grid.axes_coords
    val_r_outer = np.broadcast_to(6 * rs * np.sin(zs), (3, 32))
    bcs = [({"normal_derivative": 0}, {"normal_curvature": val_r_outer}), "periodic"]
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
