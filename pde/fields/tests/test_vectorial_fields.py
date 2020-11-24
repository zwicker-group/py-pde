"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde import CartesianGrid, ScalarField, Tensor2Field, UnitGrid, VectorField
from pde.fields.base import FieldBase
from pde.tools.misc import module_available, skipUnlessModule


def test_vectors():
    """ test some vector fields """
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])
    v1 = VectorField(grid, np.full((2,) + grid.shape, 1))
    v2 = VectorField(grid, np.full((2,) + grid.shape, 2))
    np.testing.assert_allclose(v1.average, (1, 1))
    assert np.allclose(v1.magnitude, np.sqrt(2))

    v3 = v1 + v2
    assert v3.grid == grid
    np.testing.assert_allclose(v3.data, 3)
    v1 += v2
    np.testing.assert_allclose(v1.data, 3)

    # test projections
    v1 = VectorField(grid)
    v1.data[0, :] = 3
    v1.data[1, :] = 4
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
    dot_op = v1.make_dot_operator()
    res = ScalarField(grid, dot_op(v1.data, v2.data))
    for s in (v1 @ v2, v2 @ v1, v1.dot(v2), res):
        assert isinstance(s, ScalarField)
        assert s.grid is grid
        np.testing.assert_allclose(s.data, np.full(grid.shape, 4))

    # test options for plotting images
    if module_available("matplotlib"):
        v1.plot(method="streamplot", transpose=True)


def test_divergence():
    """ test the divergence operator """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(x) + y, np.sin(y) - x]
    v = VectorField(grid, data)

    s1 = v.divergence("natural")
    assert s1.data.shape == (16, 16)
    div = np.cos(y) - np.sin(x)
    np.testing.assert_allclose(s1.data, div, rtol=0.1, atol=0.1)

    v.divergence("natural", out=s1)
    assert s1.data.shape == (16, 16)
    np.testing.assert_allclose(s1.data, div, rtol=0.1, atol=0.1)


def test_vector_gradient():
    """ test the vector gradient operator """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(x) + y, np.sin(y) - x]
    v = VectorField(grid, data)

    t1 = v.gradient("periodic")
    assert t1.data.shape == (2, 2, 16, 16)
    d00 = -np.sin(x)
    d10 = np.ones(grid.shape)
    d01 = -d10.copy()
    d10[:, 0] = d10[:, -1] = -7
    d01[0, :] = d01[-1, :] = 7
    d11 = np.cos(y)
    t2 = Tensor2Field(grid, np.array([[d00, d01], [d10, d11]]))
    np.testing.assert_allclose(t1.data, t2.data, rtol=0.1, atol=0.1)

    v.gradient("natural", out=t1)
    assert t1.data.shape == (2, 2, 16, 16)
    np.testing.assert_allclose(t1.data, t2.data, rtol=0.1, atol=0.1)


def test_vector_laplace():
    """ test the laplace operator """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = [np.cos(x) + np.sin(y), np.sin(y) - np.cos(x)]
    v = VectorField(grid, data)
    vl = v.laplace("natural")
    assert vl.data.shape == (2, 16, 16)
    np.testing.assert_allclose(
        vl.data[0, ...], -np.cos(x) - np.sin(y), rtol=0.1, atol=0.1
    )
    np.testing.assert_allclose(
        vl.data[1, ...], -np.sin(y) + np.cos(x), rtol=0.1, atol=0.1
    )


def test_vector_boundary_conditions():
    """ test some boundary conditions of operators of vector fields """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 1]], 32, periodic=False)
    v_x = np.sin(grid.cell_coords[..., 0])
    v_y = grid.cell_coords[..., 1]
    vf = VectorField(grid, np.array([v_x, v_y]))

    bc_x = [
        {"type": "derivative", "value": [0, -1]},
        {"type": "derivative", "value": [0, 1]},
    ]
    bc_y = [{"type": "value", "value": [0, 0]}, {"type": "value", "value": [1, 1]}]
    tf = vf.gradient(bc=[bc_x, bc_y])

    np.testing.assert_allclose(tf[0, 1].data[1:-1, :], 0)
    np.testing.assert_allclose(tf[1, 1].data, 1)


def test_outer_product():
    """ test outer product of vector fields """
    vf = VectorField(UnitGrid([1, 1]), [[[1]], [[2]]])
    outer = vf.make_outer_prod_operator()

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
    """ test initializing vector fields with expressions """
    grid = UnitGrid([4, 4])
    vf = VectorField.from_expression(grid, ["x**2", "x * y"])
    xs = grid.cell_coords[..., 0]
    ys = grid.cell_coords[..., 1]
    np.testing.assert_allclose(vf.data[0], xs ** 2)
    np.testing.assert_allclose(vf.data[1], xs * ys)

    # corner case
    vf = VectorField.from_expression(grid, ["1", "x * y"])

    with pytest.raises(ValueError):
        VectorField.from_expression(grid, "xy")
    with pytest.raises(ValueError):
        VectorField.from_expression(grid, ["xy"])
    with pytest.raises(ValueError):
        VectorField.from_expression(grid, ["x"] * 3)


def test_vector_plot_quiver_reduction():
    """ test whether quiver plots reduce the resolution """
    grid = UnitGrid([6, 6])
    field = VectorField.random_normal(grid)
    ref = field.plot(method="quiver", max_points=4)
    assert len(ref.element.U) == 16


def test_boundary_interpolation_vector():
    """ test boundary interpolation """
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 3])
    field = VectorField.random_normal(grid)

    # test boundary interpolation
    bndry_val = np.random.randn(2, 3)
    for bndry in grid._iter_boundaries():
        val = field.get_boundary_values(*bndry, bc={"value": bndry_val})
        np.testing.assert_allclose(val, bndry_val)

        ev = field.make_get_boundary_values(*bndry, bc={"value": bndry_val})
        np.testing.assert_allclose(ev(), bndry_val)


def test_plotting_2d():
    """ test plotting of 2d vector fields """
    grid = UnitGrid([3, 3])
    field = VectorField.random_uniform(grid, 0.1, 0.9)

    for method in ["quiver", "streamplot"]:
        ref = field.plot(method=method)
        field._update_plot(ref)


@skipUnlessModule("napari")
def test_interactive_vector_plotting():
    """ test the interactive plotting """
    grid = UnitGrid([3, 3])
    field = VectorField.random_uniform(grid, 0.1, 0.9)
    field.plot_interactive(viewer_args={"show": False, "close": True})


def test_complex_vectors():
    """ test some complex vector fields """
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])
    shape = (2, 2) + grid.shape
    numbers = np.random.random(shape) + np.random.random(shape) * 1j
    v1 = VectorField(grid, numbers[0])
    v2 = VectorField(grid, numbers[1])
    assert v1.is_complex and v2.is_complex
    dot_op = v1.make_dot_operator()

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
    dot_op = v1.make_dot_operator(conjugate=False)
    res = v1.dot(v2, conjugate=False)
    np.testing.assert_allclose(dot_op(v1.data, v2.data), res.data)
