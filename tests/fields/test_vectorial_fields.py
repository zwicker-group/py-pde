"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, ScalarField, Tensor2Field, UnitGrid, VectorField
from pde.fields.base import FieldBase
from pde.grids import CylindricalSymGrid, PolarSymGrid, SphericalSymGrid
from pde.tools.misc import module_available


def test_vectors_basic():
    """Test some vector fields."""
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])
    v1 = VectorField(grid, np.full((2,) + grid.shape, 1))
    v2 = VectorField(grid, np.full((2,) + grid.shape, 2))
    np.testing.assert_allclose(v1.average, (1, 1))
    assert np.allclose(v1.magnitude, np.sqrt(2))

    assert v1[0] == v1["x"]
    assert v1[1] == v1["y"]
    v1[0] = v1[0]
    with pytest.raises(IndexError):
        v1["z"]

    v3 = v1 + v2
    assert v3.grid == grid
    np.testing.assert_allclose(v3.data, 3)
    v1 += v2
    np.testing.assert_allclose(v1.data, 3)

    # test projections
    v1 = VectorField(grid)
    v1[0] = 3
    v1[1] = 4
    for method, value in [
        ("min", 3),
        ("max", 4),
        ("norm", 5),
        ("squared_sum", 25),
        ("norm_squared", 25),
        ("auto", 5),
    ]:
        p1 = v1.to_scalar(method)
        assert p1.data.shape == grid.shape
        np.testing.assert_allclose(p1.data, value)

    v2 = FieldBase.from_state(v1.attributes, data=v1.data)
    assert v1 == v2
    assert v1.grid is v2.grid

    attrs = VectorField.unserialize_attributes(v1.attributes_serialized)
    v2 = FieldBase.from_state(attrs, data=v1.data)
    assert v1 == v2
    assert v1.grid is not v2.grid

    # test dot product
    v2._grid = v1.grid  # make sure grids are identical
    v1.data = 1
    v2.data = 2

    for backend in ["numpy", "numba"]:
        dot_op = v1.make_dot_operator(backend)
        res = ScalarField(grid, dot_op(v1.data, v2.data))
        for s in (v1 @ v2, v2 @ v1, v1.dot(v2), res):
            assert isinstance(s, ScalarField)
            assert s.grid is grid
            np.testing.assert_allclose(s.data, np.full(grid.shape, 4))

    # test options for plotting images
    if module_available("matplotlib"):
        v1.plot(method="streamplot", transpose=True)


def test_divergence():
    """Test the divergence operator."""
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(x) + y, np.sin(y) - x]
    v = VectorField(grid, data)

    s1 = v.divergence("auto_periodic_neumann")
    assert s1.data.shape == (16, 16)
    div = np.cos(y) - np.sin(x)
    np.testing.assert_allclose(s1.data, div, rtol=0.1, atol=0.1)

    v.divergence("auto_periodic_neumann", out=s1)
    assert s1.data.shape == (16, 16)
    np.testing.assert_allclose(s1.data, div, rtol=0.1, atol=0.1)


def test_vector_gradient_field():
    """Test the vector gradient operator."""
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(x) + y, np.sin(y) - x]
    v = VectorField(grid, data)

    t1 = v.gradient("periodic")
    assert t1.data.shape == (2, 2, 16, 16)
    d00 = -np.sin(x)
    d10 = -np.ones(grid.shape)
    d01 = np.ones(grid.shape)
    d11 = np.cos(y)
    t2 = Tensor2Field(grid, np.array([[d00, d01], [d10, d11]]))
    np.testing.assert_allclose(
        t1.data[1:-1, 1:-1], t2.data[1:-1, 1:-1], rtol=0.1, atol=0.1
    )

    v.gradient("auto_periodic_neumann", out=t1)
    assert t1.data.shape == (2, 2, 16, 16)
    np.testing.assert_allclose(
        t1.data[1:-1, 1:-1], t2.data[1:-1, 1:-1], rtol=0.1, atol=0.1
    )


def test_vector_laplace():
    """Test the laplace operator."""
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(x) + np.sin(y), np.sin(y) - np.cos(x)]
    v = VectorField(grid, data)
    vl = v.laplace("auto_periodic_neumann")
    assert vl.data.shape == (2, 16, 16)
    np.testing.assert_allclose(
        vl.data[0, ...], -np.cos(x) - np.sin(y), rtol=0.1, atol=0.1
    )
    np.testing.assert_allclose(
        vl.data[1, ...], -np.sin(y) + np.cos(x), rtol=0.1, atol=0.1
    )


def test_vector_boundary_conditions():
    """Test some boundary conditions of operators of vector fields."""
    grid = CartesianGrid([[0, 2 * np.pi], [0, 1]], 32, periodic=[False, True])
    vf = VectorField.from_expression(grid, ["sin(x)", "0"])

    bc = {"x-": {"derivative": [-1, 0]}, "x+": {"derivative": [1, 0]}, "y": "periodic"}
    tf = vf.gradient(bc=bc)

    res = ScalarField.from_expression(grid, "cos(x)")
    np.testing.assert_allclose(tf[0, 0].data, res.data, atol=0.01, rtol=0.01)
    np.testing.assert_allclose(tf[0, 1].data, 0)
    np.testing.assert_allclose(tf[1, 0].data, 0)
    np.testing.assert_allclose(tf[1, 1].data, 0)


def test_outer_product():
    """Test outer product of vector fields."""
    vf = VectorField(UnitGrid([1, 1]), [[[1]], [[2]]])

    for backend in ["numpy", "numba"]:
        outer = vf.make_outer_prod_operator(backend)

        tf = vf.outer_product(vf)
        res = np.array([1, 2, 2, 4]).reshape(2, 2, 1, 1)
        np.testing.assert_equal(tf.data, res)
        np.testing.assert_equal(outer(vf.data, vf.data), res)

        tf.data = 0
        res = np.array([1, 2, 2, 4]).reshape(2, 2, 1, 1)
        vf.outer_product(vf, out=tf)
        np.testing.assert_equal(tf.data, res)
        outer(vf.data, vf.data, out=tf.data)
        np.testing.assert_equal(tf.data, res)


def test_from_expressions():
    """Test initializing vector fields with expressions."""
    grid = UnitGrid([4, 4])
    vf = VectorField.from_expression(grid, ["x**2", "x * y"])
    xs = grid.cell_coords[..., 0]
    ys = grid.cell_coords[..., 1]
    np.testing.assert_allclose(vf.data[0], xs**2)
    np.testing.assert_allclose(vf.data[1], xs * ys)

    vf2 = VectorField.from_expression(grid, ["cartesian[0]**2", "x * y"])
    np.testing.assert_array_equal(vf.data, vf2.data)
    vf3 = VectorField.from_expression(grid, ["x**2", "x * cartesian[1]"])
    np.testing.assert_array_equal(vf.data, vf3.data)

    # corner case
    vf = VectorField.from_expression(grid, ["1", "x * y"])
    np.testing.assert_allclose(vf.data[0], 1)
    vf = VectorField.from_expression(grid, [1, "x * y"])
    np.testing.assert_allclose(vf.data[0], 1)

    with pytest.raises(ValueError):
        VectorField.from_expression(grid, "xy")
    with pytest.raises(ValueError):
        VectorField.from_expression(grid, ["xy"])
    with pytest.raises(ValueError):
        VectorField.from_expression(grid, ["x"] * 3)

    # user functions and constants
    def f(x, y):
        return x**2

    vf = VectorField.from_expression(
        grid,
        ["f(x, y)", "xy"],
        user_funcs={"f": f},
        consts={"xy": np.outer(*grid.axes_coords)},
    )
    np.testing.assert_allclose(vf.data[0], xs**2)
    np.testing.assert_allclose(vf.data[1], xs * ys)


def test_vector_plot_quiver_reduction(rng):
    """Test whether quiver plots reduce the resolution."""
    grid = UnitGrid([6, 6])
    field = VectorField.random_normal(grid, rng=rng)
    ref = field.plot(method="quiver", max_points=4)
    assert len(ref.element.U) == 16


def test_boundary_interpolation_vector(rng):
    """Test boundary interpolation."""
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 3])
    field = VectorField.random_normal(grid, rng=rng)

    # test boundary interpolation
    bndry_val = rng.normal(size=(2, 3))
    for bndry in grid._iter_boundaries():
        val = field.get_boundary_values(*bndry, bc={"value": bndry_val})
        np.testing.assert_allclose(val, bndry_val)


@pytest.mark.parametrize("transpose", [True, False])
def test_vector_plotting_2d(transpose, rng):
    """Test plotting of 2d vector fields."""
    grid = UnitGrid([3, 4])
    field = VectorField.random_uniform(grid, 0.1, 0.9, rng=rng)

    for method in ["quiver", "streamplot"]:
        ref = field.plot(method=method, transpose=transpose)
        field._update_plot(ref)

    # test sub-sampling
    grid = UnitGrid([32, 15])
    field = VectorField.random_uniform(grid, 0.1, 0.9, rng=rng)
    field.get_vector_data(transpose=transpose, max_points=7)


@pytest.mark.skipif(not module_available("napari"), reason="requires `napari` module")
@pytest.mark.interactive
def test_interactive_vector_plotting(rng):
    """Test the interactive plotting."""
    grid = UnitGrid([3, 3])
    field = VectorField.random_uniform(grid, 0.1, 0.9, rng=rng)
    field.plot_interactive(viewer_args={"show": False, "close": True})


def test_complex_vectors(rng):
    """Test some complex vector fields."""
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])
    shape = (2, 2) + grid.shape
    numbers = rng.random(shape) + rng.random(shape) * 1j
    v1 = VectorField(grid, numbers[0])
    v2 = VectorField(grid, numbers[1])
    assert v1.is_complex
    assert v2.is_complex

    for backend in ["numpy", "numba"]:
        dot_op = v1.make_dot_operator(backend)

        # test complex conjugate
        expected = v1.to_scalar("norm_squared").data
        np.testing.assert_allclose((v1 @ v1).data, expected)
        np.testing.assert_allclose(dot_op(v1.data, v1.data), expected)

        # test dot product
        res = dot_op(v1.data, v2.data)
        for s in (v1 @ v2, (v2 @ v1).conjugate(), v1.dot(v2)):
            assert isinstance(s, ScalarField)
            assert s.grid is grid
            np.testing.assert_allclose(s.data, res)

        # test without conjugate
        dot_op = v1.make_dot_operator(backend, conjugate=False)
        res = v1.dot(v2, conjugate=False)
        np.testing.assert_allclose(dot_op(v1.data, v2.data), res.data)


def test_vector_bcs():
    """Test boundary conditions on vector fields."""
    grid = UnitGrid([3, 3], periodic=False)
    v = VectorField.from_expression(grid, ["x", "cos(y)"])

    bcs = {"x": {"value": [0, 1]}, "y": {"value": [2, 3]}}
    s1 = v.divergence(bcs, backend="scipy").data
    div = grid.make_operator("divergence", bcs)
    s2 = div(v.data)

    np.testing.assert_allclose(s1, s2)


def test_interpolation_vector_fields_cartesian():
    """Test interpolation of a vector field on Cartesian coordinates."""
    # upscale
    grid = UnitGrid([10, 10])
    vf = VectorField.from_expression(grid, ["x", "y**2"])
    grid2 = CartesianGrid([[0, 10], [0, 10]], 20)
    vf2 = vf.interpolate_to_grid(grid2)
    vf_expect = VectorField.from_expression(grid2, ["x", "y**2"])
    np.testing.assert_allclose(vf2.data[1:-1, 1:-1], vf_expect.data[1:-1, 1:-1])

    # downscale
    grid = CartesianGrid([[0, 10], [0, 10]], 20)
    vf = VectorField.from_expression(grid, ["x", "y**2"])
    grid2 = UnitGrid([10, 10])
    vf2 = vf.interpolate_to_grid(grid2)
    vf_expect = VectorField.from_expression(grid2, ["x", "y**2"])
    np.testing.assert_allclose(vf2.data[1:-1, 1:-1], vf_expect.data[1:-1, 1:-1])


def test_interpolation_vector_fields_cylindrical():
    """Test interpolation of a vector field on cylindrical coordinates."""
    grid = CylindricalSymGrid(5, [-2, 3], 10)
    vf = VectorField.from_expression(grid, ["r", "1", "z"])
    grid_cart = grid.get_cartesian_grid(mode="valid")
    vf_cart = vf.interpolate_to_grid(grid_cart)

    x, y, z = grid_cart.coordinate_arrays
    r = np.hypot(x, y)
    np.testing.assert_allclose(vf_cart.data[0], x - y / r)
    np.testing.assert_allclose(vf_cart.data[1], y + x / r)
    np.testing.assert_allclose(vf_cart.data[2], z)

    vf2 = vf.interpolate_to_grid(grid)
    np.testing.assert_allclose(vf.data, vf2.data)


def test_interpolation_vector_fields_polar():
    """Test interpolation of a vector field on polar coordinates."""
    grid = PolarSymGrid(5, 10)
    vf = VectorField.from_expression(grid, ["r", "1"])
    grid_cart = grid.get_cartesian_grid(mode="valid")
    vf_cart = vf.interpolate_to_grid(grid_cart)

    x, y = grid_cart.coordinate_arrays
    r = np.hypot(x, y)
    np.testing.assert_allclose(vf_cart.data[0], x - y / r)
    np.testing.assert_allclose(vf_cart.data[1], y + x / r)

    vf2 = vf.interpolate_to_grid(grid)
    np.testing.assert_allclose(vf.data, vf2.data)


def test_interpolation_vector_fields_spherical():
    """Test interpolation of a vector field on polar coordinates."""
    grid = SphericalSymGrid(5, 10)
    vf = VectorField.from_expression(grid, ["r", "1", "r"])
    grid_cart = grid.get_cartesian_grid(mode="valid")
    vf_cart = vf.interpolate_to_grid(grid_cart)

    x, y, z = grid_cart.coordinate_arrays
    d = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    np.testing.assert_allclose(vf_cart.data[0], x + x * z / d / r - y * r / d)
    np.testing.assert_allclose(vf_cart.data[1], y + y * z / d / r + x * r / d)
    np.testing.assert_allclose(vf_cart.data[2], z - d / r)

    vf2 = vf.interpolate_to_grid(grid)
    np.testing.assert_allclose(vf.data, vf2.data)
