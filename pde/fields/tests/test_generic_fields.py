"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import ndimage

from pde.fields import FieldCollection, ScalarField, Tensor2Field, VectorField
from pde.fields.base import DataFieldBase, FieldBase
from pde.fields.tests.fixtures import iter_grids
from pde.grids import (
    CartesianGrid,
    CylindricalSymGrid,
    PolarSymGrid,
    SphericalSymGrid,
    UnitGrid,
)
from pde.tools.misc import skipUnlessModule


@pytest.mark.parametrize("field_class", [ScalarField, VectorField, Tensor2Field])
def test_set_label(field_class):
    """test setting the field labels"""
    grid = UnitGrid([2])
    assert field_class(grid).label is None
    f = field_class(grid, label="a")
    assert f.label == "a"
    f.label = "b"
    assert f.label == "b"
    f.label = None
    assert f.label is None

    with pytest.raises(TypeError):
        f.label = 3
    with pytest.raises(TypeError):
        field_class(grid, label=1)


@pytest.mark.parametrize("grid", iter_grids())
@pytest.mark.parametrize("field_class", [ScalarField, Tensor2Field])
def test_interpolation_natural(grid, field_class):
    """test some interpolation for natural boundary conditions"""
    msg = f"grid={grid}, field={field_class}"
    f = field_class.random_uniform(grid)

    def get_point():
        if isinstance(grid, CartesianGrid):
            return grid.get_random_point(boundary_distance=0.5, coords="grid")
        else:
            return grid.get_random_point(
                boundary_distance=1, avoid_center=True, coords="grid"
            )

    # interpolate at cell center
    c = (1,) * len(grid.axes)
    p = f.grid.cell_coords[c]
    val = f.interpolate(p)
    np.testing.assert_allclose(val, f.data[(Ellipsis,) + c], err_msg=msg)


@pytest.mark.parametrize("num", [1, 3])
@pytest.mark.parametrize("grid", iter_grids())
def test_shapes_nfields(num, grid):
    """test single component field"""
    fields = [ScalarField.random_uniform(grid) for _ in range(num)]
    field = FieldCollection(fields)
    data_shape = (num,) + grid.shape
    np.testing.assert_equal(field.data.shape, data_shape)
    for pf_single in field:
        np.testing.assert_equal(pf_single.data.shape, grid.shape)

    field_c = field.copy()
    np.testing.assert_allclose(field.data, field_c.data)
    assert field.grid == field_c.grid


def test_arithmetics():
    """test simple arithmetics for fields"""
    grid = UnitGrid([2, 2])
    for cls in (ScalarField, VectorField, Tensor2Field):
        f1 = cls(grid, data=1)
        f2 = cls(grid, data=2)
        assert isinstance(str(f1), str)
        np.testing.assert_allclose(f1.data, 1)

        np.testing.assert_allclose((-f1).data, -1)

        # test addition
        np.testing.assert_allclose((f1 + 1).data, 2)
        np.testing.assert_allclose((1 + f1).data, 2)
        f1 += 1
        np.testing.assert_allclose(f1.data, 2)
        np.testing.assert_allclose((f1 + f2).data, 4)

        # test subtraction
        np.testing.assert_allclose((f1 - 1).data, 1)
        np.testing.assert_allclose((1 - f1).data, -1)
        f1 -= 1
        np.testing.assert_allclose(f1.data, 1)
        np.testing.assert_allclose((f1 - f2).data, -1)

        # test multiplication
        np.testing.assert_allclose((f1 * 2).data, 2)
        np.testing.assert_allclose((2 * f1).data, 2)
        f1 *= 2
        np.testing.assert_allclose(f1.data, 2)

        # test division
        np.testing.assert_allclose((f1 / 2).data, 1)
        f1.data = 4
        np.testing.assert_allclose((2 / f1).data, 0.5)
        f1 /= 2
        np.testing.assert_allclose(f1.data, 2)

        # test power
        f1.data = 2
        np.testing.assert_allclose((f1**3).data, 8)
        f1 **= 3
        np.testing.assert_allclose(f1.data, 8)

        # test applying a function
        f1.data = 2
        np.testing.assert_allclose(f1.apply(lambda x: x**3).data, 8)
        f1.apply(lambda x: x**3, out=f1)
        np.testing.assert_allclose(f1.data, 8)


def test_scalar_arithmetics():
    """test simple arithmetics involving scalar fields"""
    grid = UnitGrid([3, 4])
    s = ScalarField(grid, data=2)
    v = VectorField.random_uniform(grid)

    for f in [v, FieldCollection([v])]:
        f.data = s
        assert f.data.shape == (2, 3, 4)
        np.testing.assert_allclose(f.data, 2)

        f += s
        np.testing.assert_allclose(f.data, 4)
        np.testing.assert_allclose((f + s).data, 6)
        np.testing.assert_allclose((s + f).data, 6)
        f -= s
        np.testing.assert_allclose((f - s).data, 0)
        np.testing.assert_allclose((s - f).data, 0)

        f *= s
        np.testing.assert_allclose(f.data, 4)
        np.testing.assert_allclose((f * s).data, 8)
        np.testing.assert_allclose((s * f).data, 8)
        f /= s
        np.testing.assert_allclose((f / s).data, 1)
        with pytest.raises(TypeError):
            s / f
        with pytest.raises(TypeError):
            s /= f
        with pytest.raises(TypeError):
            s *= f


def test_data_managment():
    """test how data is set"""
    grid = UnitGrid([2, 2])
    for cls in (ScalarField, VectorField, Tensor2Field):
        s1 = cls(grid, data=1)
        np.testing.assert_allclose(s1.data, 1)

        s2 = cls(grid)
        np.testing.assert_allclose(s2.data, 0)

        c = FieldCollection([s1, s2])
        s1.data = 0
        np.testing.assert_allclose(c.data, 0)

        c.data = 2
        np.testing.assert_allclose(s1.data, 2)
        np.testing.assert_allclose(s2.data, 2)

        c.data += 1
        np.testing.assert_allclose(s1.data, 3)
        np.testing.assert_allclose(s2.data, 3)

        c[0].data += 2  # reference to s1
        c[1].data *= 2  # reference to s2
        np.testing.assert_allclose(s1.data, 5)
        np.testing.assert_allclose(s2.data, 6)

        c[0] = s2
        np.testing.assert_allclose(c.data, 6)

        # nested collections
        with pytest.raises(RuntimeError):
            FieldCollection([c])


@pytest.mark.parametrize("field_class", [ScalarField, VectorField, Tensor2Field])
def test_complex_fields(field_class):
    """test operations on complex fields"""
    grid = UnitGrid([3])

    field = field_class.random_uniform(grid, 0, 1 + 1j)
    assert field.is_complex
    assert field.dtype == np.dtype("complex")

    field_copy = field.copy()
    assert field_copy.is_complex
    assert field_copy.dtype == np.dtype("complex")


@skipUnlessModule("h5py")
def test_hdf_input_output(tmp_path):
    """test writing and reading files"""
    grid = UnitGrid([4, 4])
    s = ScalarField.random_uniform(grid, label="scalar")
    v = VectorField.random_uniform(grid, label="vector")
    t = Tensor2Field.random_uniform(grid, label="tensor")
    col = FieldCollection([s, v, t], label="collection")

    path = tmp_path / "test_hdf_input_output.hdf5"
    for f in [s, v, t, col]:
        f.to_file(path)
        f2 = FieldBase.from_file(path)
        assert f == f2
        assert f.label == f2.label
        assert isinstance(str(f), str)
        assert isinstance(repr(f), str)


def test_writing_images(tmp_path):
    """test writing and reading files"""
    from matplotlib.pyplot import imread

    grid = UnitGrid([4, 4])
    s = ScalarField.random_uniform(grid, label="scalar")
    v = VectorField.random_uniform(grid, label="vector")
    t = Tensor2Field.random_uniform(grid, label="tensor")

    path = tmp_path / "test_writing_images.png"
    for f in [s, v, t]:
        f.to_file(path)
        # try reading the file
        with path.open("br") as fp:
            imread(fp)


@pytest.mark.parametrize("ndim", [1, 2])
def test_interpolation_to_grid_fields(ndim):
    """test whether data is interpolated correctly for different fields"""
    grid = CartesianGrid([[0, 2 * np.pi]] * ndim, 6)
    grid2 = CartesianGrid([[0, 2 * np.pi]] * ndim, 8)
    if ndim == 1:
        vf = VectorField.from_expression(grid, ["cos(x)"])
    elif ndim == 2:
        vf = VectorField.from_expression(grid, ["sin(y)", "cos(x)"])
    sf = vf[0]  # test extraction of fields
    fc = FieldCollection([sf, vf])

    for f in [sf, vf, fc]:
        # test self-interpolation
        f0 = f.interpolate_to_grid(grid)
        np.testing.assert_allclose(f.data, f0.data, atol=1e-15)

        # test interpolation to finer grid and back
        f2 = f.interpolate_to_grid(grid2)
        f3 = f2.interpolate_to_grid(grid)
        np.testing.assert_allclose(f.data, f3.data, atol=0.2, rtol=0.2)


@pytest.mark.parametrize("field_cls", [ScalarField, VectorField, Tensor2Field])
def test_interpolation_values(field_cls):
    """test whether data is interpolated correctly for different fields"""
    grid = UnitGrid([3, 4])
    f = field_cls.random_uniform(grid)
    f.set_ghost_cells("auto_periodic_neumann")

    intp = f.make_interpolator()
    c = f.grid.cell_coords[2, 2]
    np.testing.assert_allclose(intp(c), f.data[..., 2, 2])

    intp = f.make_interpolator(with_ghost_cells=True)
    c = f.grid.cell_coords[2, 2]
    np.testing.assert_allclose(intp(c, f._data_full), f.data[..., 2, 2])

    with pytest.raises((ValueError, IndexError)):
        intp(np.array([100, -100]))

    res = f.make_interpolator(fill=45)(np.array([100, -100]))
    np.testing.assert_almost_equal(res, np.full(f.data_shape, 45))


def test_interpolation_ghost_cells():
    """test whether data is interpolated correctly with or without ghost cells"""
    grid = UnitGrid([3])
    f = ScalarField(grid, [1, 2, 3])
    f.set_ghost_cells({"value": 0})

    intp = f.make_interpolator()
    res = intp(np.c_[0:4])
    np.testing.assert_allclose(res, np.array([1.0, 1.5, 2.5, 3.0]))

    intp = f.make_interpolator(with_ghost_cells=True)
    res = intp(np.c_[0:4])
    np.testing.assert_allclose(res, np.array([0, 1.5, 2.5, 0]))

    grid = UnitGrid([3], periodic=True)
    f = ScalarField(grid, [1, 2, 3])
    f.set_ghost_cells("periodic")

    intp = f.make_interpolator()
    res = intp(np.c_[0:4])
    np.testing.assert_allclose(res, np.array([2.0, 1.5, 2.5, 2.0]))

    intp = f.make_interpolator(with_ghost_cells=True)
    res = intp(np.c_[0:4])
    np.testing.assert_allclose(res, np.array([2.0, 1.5, 2.5, 2.0]))


@pytest.mark.parametrize(
    "grid",
    [
        UnitGrid((6,)),
        PolarSymGrid(6, 4),
        SphericalSymGrid(7, 4),
        CylindricalSymGrid(6, (0, 8), (7, 8)),
    ],
)
def test_interpolation_to_cartesian(grid):
    """test whether data is interpolated correctly to Cartesian grid"""
    dim = grid.dim
    vf = VectorField(grid, 2)
    sf = vf[0]  # test extraction of fields
    fc = FieldCollection([sf, vf])

    # subset
    grid_cart = UnitGrid([4] * dim)
    for f in [sf, fc]:
        res = f.interpolate_to_grid(grid_cart)
        np.testing.assert_allclose(res.data, 2)

    # superset
    grid_cart = UnitGrid([8] * dim)
    for f in [sf, fc]:
        res = f.interpolate_to_grid(grid_cart, fill=0)
        assert res.data.min() == 0
        assert res.data.max() == pytest.approx(2)


@pytest.mark.parametrize(
    "grid",
    [PolarSymGrid(6, 4), SphericalSymGrid(7, 4), CylindricalSymGrid(6, (0, 8), (7, 8))],
)
def test_get_cartesian_grid(grid):
    """test whether Cartesian grids can be created"""
    cart = grid.get_cartesian_grid(mode="valid")
    assert cart.volume < grid.volume
    cart = grid.get_cartesian_grid(mode="full")
    assert cart.volume > grid.volume


@pytest.mark.parametrize("grid", iter_grids())
def test_simple_plotting(grid):
    """test simple plotting of various fields on various grids"""
    vf = VectorField.random_uniform(grid)
    tf = Tensor2Field.random_uniform(grid)
    sf = tf[0, 0]  # test extraction of fields
    fc = FieldCollection([sf, vf])
    for f in [sf, vf, tf, fc]:
        f.plot(action="close")
        f.plot(kind="line", action="close")
        if grid.dim >= 2:
            f.plot(kind="image", action="close")
        if isinstance(f, VectorField) and grid.dim == 2:
            f.plot(kind="quiver", action="close")
            f.plot(kind="streamplot", action="close")


def test_random_uniform():
    """test whether random uniform fields behave correctly"""
    grid = UnitGrid([256, 256])
    for field_cls in [ScalarField, VectorField, Tensor2Field]:
        a = np.random.random()
        b = 2 + np.random.random()
        f = field_cls.random_uniform(grid, a, b)
        assert np.mean(f.average) == pytest.approx((a + b) / 2, rel=0.02)
        assert np.std(f.data) == pytest.approx(0.288675 * (b - a), rel=0.1)

        np.testing.assert_allclose(f.real.data, f.data)
        np.testing.assert_allclose(f.imag.data, 0)


def test_random_uniform_types():
    """test whether random uniform fields behave correctly for different types"""
    grid = UnitGrid([8])
    for dtype in [bool, int, float, complex]:
        field = VectorField.random_uniform(grid, dtype=dtype)
        assert field.dtype == np.dtype(dtype)
        assert isinstance(field.data.flat[0].item(), dtype)

    assert ScalarField.random_uniform(grid, 0, 1).dtype == np.dtype(float)
    assert ScalarField.random_uniform(grid, vmin=0 + 0j).dtype == np.dtype(complex)
    assert ScalarField.random_uniform(grid, vmax=1 + 0j).dtype == np.dtype(complex)
    assert ScalarField.random_uniform(grid, 0 + 0j, 1 + 0j).dtype == np.dtype(complex)


def test_random_normal():
    """test whether random normal fields behave correctly"""
    grid = UnitGrid([256, 256])
    for field_cls in [ScalarField, VectorField, Tensor2Field]:
        m = np.random.random()
        s = 1 + np.random.random()
        for scaling in ["none", "physical"]:
            f = field_cls.random_normal(grid, mean=m, std=s, scaling=scaling)
            assert np.mean(f.average) == pytest.approx(m, rel=0.1, abs=0.1)
            assert np.std(f.data) == pytest.approx(s, rel=0.1, abs=0.1)


def test_random_normal_types():
    """test whether random normal fields behave correctly for different types"""
    grid = UnitGrid([8])
    for dtype in [bool, int, float, complex]:
        field = VectorField.random_normal(grid, dtype=dtype)
        assert field.dtype == np.dtype(dtype)
        assert isinstance(field.data.flat[0].item(), dtype)

    assert ScalarField.random_normal(grid, 0, 1).dtype == np.dtype(float)
    assert ScalarField.random_normal(grid, mean=0 + 0j).dtype == np.dtype(complex)
    assert ScalarField.random_normal(grid, std=1 + 0j).dtype == np.dtype(complex)
    assert ScalarField.random_normal(grid, 0 + 0j, 1 + 0j).dtype == np.dtype(complex)

    m = complex(np.random.random(), np.random.random())
    s = complex(1 + np.random.random(), 1 + np.random.random())
    grid = UnitGrid([256, 256])
    field = field.random_normal(grid, m, s)
    assert np.mean(field.average) == pytest.approx(m, rel=0.1, abs=0.1)
    assert np.std(field.data.real) == pytest.approx(s.real, rel=0.1, abs=0.1)
    assert np.std(field.data.imag) == pytest.approx(s.imag, rel=0.1, abs=0.1)


@pytest.mark.parametrize("field_cls", [ScalarField, VectorField, Tensor2Field])
def test_random_colored(field_cls):
    """test whether random colored fields behave correctly"""
    grid = UnitGrid([128, 128])
    exponent = np.random.uniform(-4, 4)
    scale = 1 + np.random.random()
    f = field_cls.random_colored(grid, exponent=exponent, scale=scale)

    assert np.allclose(f.average, 0)


def test_random_rng():
    """test whether the random number generator arguments are accepted"""
    grid = UnitGrid([2, 2])
    for create_random_field in [
        ScalarField.random_colored,
        ScalarField.random_harmonic,
        ScalarField.random_normal,
        ScalarField.random_uniform,
    ]:
        f1 = create_random_field(grid, rng=np.random.default_rng(1))
        f2 = create_random_field(grid, rng=np.random.default_rng(1))
        np.testing.assert_allclose(f1.data, f2.data)


def test_fluctuations():
    """test the scaling of fluctuations"""
    for dim in [1, 2]:
        for size in [256, 512]:
            if dim == 1:
                size **= 2
            grid = CartesianGrid([[0, 1]] * dim, [size] * dim)
            std = 1 + np.random.random()
            for field_cls in [ScalarField, VectorField, Tensor2Field]:
                s = field_cls.random_normal(
                    grid, mean=np.random.random(), std=std, scaling="physical"
                )
                expect = np.full([dim] * field_cls.rank, std)
                np.testing.assert_allclose(s.fluctuations, expect, rtol=0.1)


def test_smoothing():
    """test smoothing on different grids"""
    for grid in [
        CartesianGrid([[-2, 3]], 4),
        UnitGrid(7, periodic=False),
        UnitGrid(7, periodic=True),
    ]:
        f1 = ScalarField.random_uniform(grid)
        sigma = 0.5 + np.random.random()

        # this assumes that the grid periodicity is the same for all axes
        mode = "wrap" if grid.periodic[0] else "reflect"
        s = sigma / grid.typical_discretization
        expected = ndimage.gaussian_filter(f1.data, sigma=s, mode=mode)

        out = f1.smooth(sigma)
        np.testing.assert_allclose(out.data, expected)

        out.data = 0  # reset data
        f1.smooth(sigma, out=out).data
        np.testing.assert_allclose(out.data, expected)

    # test one simple higher order smoothing
    tf = Tensor2Field.random_uniform(grid)
    assert tf.data.shape == tf.smooth(1).data.shape

    # test in-place smoothing
    g = UnitGrid([8, 8])
    f1 = ScalarField.random_normal(g)
    f2 = f1.smooth(3)
    f1.smooth(3, out=f1)
    np.testing.assert_allclose(f1.data, f2.data)


def test_vector_from_scalars():
    """test how to compile vector fields from scalar fields"""
    g = UnitGrid([1, 2])
    s1 = ScalarField(g, [[0, 1]])
    s2 = ScalarField(g, [[2, 3]])
    v = VectorField.from_scalars([s1, s2], label="test")
    assert v.label == "test"
    np.testing.assert_equal(v.data, [[[0, 1]], [[2, 3]]])

    with pytest.raises(ValueError):
        VectorField.from_scalars([s1, s2, s1])


def test_dot_product():
    """test dot products between vectors and tensors"""
    g = UnitGrid([3, 2])
    vf = VectorField.random_normal(g)
    tf = Tensor2Field.random_normal(g)
    v_dot = vf.make_dot_operator()
    t_dot = tf.make_dot_operator()

    expected = np.einsum("i...,i...->...", vf.data, vf.data)
    np.testing.assert_allclose((vf @ vf).data, expected)
    np.testing.assert_allclose(v_dot(vf.data, vf.data), expected)

    expected = np.einsum("i...,i...->...", vf.data, tf.data)
    np.testing.assert_allclose((vf @ tf).data, expected)
    np.testing.assert_allclose(v_dot(vf.data, tf.data), expected)

    expected = np.einsum("ji...,i...->j...", tf.data, vf.data)
    np.testing.assert_allclose((tf @ vf).data, expected)
    np.testing.assert_allclose(t_dot(tf.data, vf.data), expected)

    expected = np.einsum("ij...,jk...->ik...", tf.data, tf.data)
    np.testing.assert_allclose((tf @ tf).data, expected)
    np.testing.assert_allclose(t_dot(tf.data, tf.data), expected)


@pytest.mark.parametrize("grid", iter_grids())
def test_complex_operator(grid):
    """test using a complex operator on grid"""
    r = ScalarField.random_normal(grid)
    i = ScalarField.random_normal(grid)
    c = r + 1j * i
    assert c.is_complex
    assert np.iscomplexobj(c)

    c_lap = c.laplace("auto_periodic_neumann").data
    np.testing.assert_allclose(c_lap.real, r.laplace("auto_periodic_neumann").data)
    np.testing.assert_allclose(c_lap.imag, i.laplace("auto_periodic_neumann").data)


def test_get_field_class_by_rank():
    """test _get_field_class_by_rank function"""
    assert DataFieldBase.get_class_by_rank(0) is ScalarField
    assert DataFieldBase.get_class_by_rank(1) is VectorField
    assert DataFieldBase.get_class_by_rank(2) is Tensor2Field
    with pytest.raises(RuntimeError):
        DataFieldBase.get_class_by_rank(3)
