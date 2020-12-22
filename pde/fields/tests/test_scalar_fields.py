"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde.fields.base import FieldBase
from pde.fields.scalar import ScalarField
from pde.grids import CartesianGrid, PolarGrid, UnitGrid, boundaries
from pde.grids.tests.test_cartesian_grids import _get_cartesian_grid
from pde.tools.misc import module_available, skipUnlessModule


def test_interpolation_singular():
    """ test interpolation on singular dimensions """
    grid = UnitGrid([1])
    field = ScalarField(grid, data=3)

    # test constant boundary conditions
    bc = [{"type": "value", "value": 1}, {"type": "value", "value": 5}]
    x = np.linspace(0, 1, 7).reshape((7, 1))
    y = field.interpolate(x, method="numba", bc=bc)
    np.testing.assert_allclose(y, 1 + 4 * x.ravel())

    # test derivative boundary conditions
    bc = [{"type": "derivative", "value": -2}, {"type": "derivative", "value": 2}]
    x = np.linspace(0, 1, 7).reshape((7, 1))
    y = field.interpolate(x, method="numba", bc=bc)
    np.testing.assert_allclose(y, 2 + 2 * x.ravel())

    # test boundary interpolation
    for upper in [True, False]:
        val = field.get_boundary_values(axis=0, upper=upper, bc=[{"value": 1}])
        assert val == pytest.approx(1)


def test_simple_shapes(example_grid):
    """ test simple scalar fields """
    pf = ScalarField.random_uniform(example_grid)
    np.testing.assert_equal(pf.data.shape, example_grid.shape)
    pf_lap = pf.laplace("natural")
    np.testing.assert_equal(pf_lap.data.shape, example_grid.shape)
    assert isinstance(pf.integral, float)

    pf_c = pf.copy()
    np.testing.assert_allclose(pf.data, pf_c.data)
    assert pf.grid == pf_c.grid
    assert pf.data is not pf_c.data

    if module_available("matplotlib"):
        pf.plot()  # simply test whether this does not cause errors


def test_scalars():
    """ test some scalar fields """
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 4])
    s1 = ScalarField(grid, np.full(grid.shape, 1))
    s2 = ScalarField(grid, np.full(grid.shape, 2))
    assert s1.average == pytest.approx(1)
    assert s1.magnitude == pytest.approx(1)

    s3 = s1 + s2
    assert s3.grid == grid
    np.testing.assert_allclose(s3.data, 3)
    s1 += s2
    np.testing.assert_allclose(s1.data, 3)

    s2 = FieldBase.from_state(s1.attributes, data=s1.data)
    assert s1 == s2
    assert s1.grid is s2.grid

    attrs = ScalarField.unserialize_attributes(s1.attributes_serialized)
    s2 = FieldBase.from_state(attrs, data=s1.data)
    assert s1 == s2
    assert s1.grid is not s2.grid

    # test options for plotting images
    if module_available("matplotlib"):
        s1.plot(transpose=True, colorbar=True)

    s3 = ScalarField(grid, s1)
    assert s1 is not s3
    assert s1 == s3
    assert s1.grid is s3.grid

    # multiplication with numpy arrays
    arr = np.random.randn(*grid.shape)
    np.testing.assert_allclose((arr * s1).data, (s1 * arr).data)


def test_laplacian():
    """ test the gradient operator """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    s = ScalarField.random_harmonic(grid, axis_combination=np.add, modes=1)

    s_lap = s.laplace("natural")
    assert s_lap.data.shape == (16, 16)
    np.testing.assert_allclose(s_lap.data, -s.data, rtol=0.1, atol=0.1)

    s.laplace("natural", out=s_lap)
    assert s_lap.data.shape == (16, 16)
    np.testing.assert_allclose(s_lap.data, -s.data, rtol=0.1, atol=0.1)


def test_gradient():
    """ test the gradient operator """
    grid = CartesianGrid([[0, 2 * np.pi], [0, 2 * np.pi]], [16, 16], periodic=True)
    x, y = grid.cell_coords[..., 0], grid.cell_coords[..., 1]
    data = np.cos(x) + np.sin(y)

    s = ScalarField(grid, data)
    v = s.gradient("natural")
    assert v.data.shape == (2, 16, 16)
    np.testing.assert_allclose(v.data[0], -np.sin(x), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(v.data[1], np.cos(y), rtol=0.1, atol=0.1)

    s.gradient("natural", out=v)
    assert v.data.shape == (2, 16, 16)
    np.testing.assert_allclose(v.data[0], -np.sin(x), rtol=0.1, atol=0.1)
    np.testing.assert_allclose(v.data[1], np.cos(y), rtol=0.1, atol=0.1)


def test_interpolation_to_grid(example_grid):
    """ test whether data is interpolated correctly for different grids """
    sf = ScalarField.random_uniform(example_grid)
    sf2 = sf.interpolate_to_grid(example_grid, method="numba")
    np.testing.assert_allclose(sf.data, sf2.data, rtol=1e-6)


def test_add_interpolated_scalar(example_grid):
    """ test the `add_interpolated` method """
    f = ScalarField(example_grid)
    a = np.random.random()

    c = tuple(example_grid.point_to_cell(example_grid.get_random_point()))
    p = example_grid.cell_to_point(c, cartesian=False)
    f.add_interpolated(p, a)
    assert f.data[c] == pytest.approx(a / example_grid.cell_volumes[c])

    f.add_interpolated(example_grid.get_random_point(cartesian=False), a)
    assert f.integral == pytest.approx(2 * a)

    f.data = 0  # reset
    add_interpolated = example_grid.make_add_interpolated_compiled()
    c = tuple(example_grid.point_to_cell(example_grid.get_random_point()))
    p = example_grid.cell_to_point(c, cartesian=False)
    add_interpolated(f.data, p, a)
    assert f.data[c] == pytest.approx(a / example_grid.cell_volumes[c])

    add_interpolated(f.data, example_grid.get_random_point(cartesian=False), a)
    assert f.integral == pytest.approx(2 * a)


def test_add_interpolated_1d():
    """ test the `add_interpolated` method for 1d systems """
    grid = PolarGrid(3, 5)
    f = ScalarField(grid)
    g = f.copy()
    a = np.random.random()
    for r in np.linspace(0, 3, 8).reshape(8, 1):
        f.data = g.data = 0
        f.add_interpolated(r, a)
        assert f.integral == pytest.approx(a)
        grid.make_add_interpolated_compiled()(g.data, r, a)
        np.testing.assert_array_almost_equal(f.data, g.data)


def test_random_harmonic():
    """ test whether random harmonic fields behave correctly """
    grid = _get_cartesian_grid(2)  # get random Cartesian grid
    x = ScalarField.random_harmonic(grid, modes=1)
    scaling = sum((2 * np.pi / L) ** 2 for L in grid.cuboid.size)
    y = -x.laplace("natural") / scaling
    np.testing.assert_allclose(x.data, y.data, rtol=1e-2, atol=1e-2)


def test_get_line_data():
    """ test different extraction methods for line data """
    grid = UnitGrid([16, 32])
    c = ScalarField.random_harmonic(grid)

    np.testing.assert_equal(
        c.get_line_data(extract="cut_0"), c.get_line_data(extract="cut_x")
    )
    np.testing.assert_equal(
        c.get_line_data(extract="cut_1"), c.get_line_data(extract="cut_y")
    )

    for extract in ["project_0", "project_1", "project_x", "project_y"]:
        data = c.get_line_data(extract=extract)["data_y"]
        np.testing.assert_allclose(data, 0, atol=1e-14)


def test_from_expression():
    """ test creating scalar field from expression """
    grid = UnitGrid([1, 2])
    sf = ScalarField.from_expression(grid, "x * y", label="abc")
    assert sf.label == "abc"
    np.testing.assert_allclose(sf.data, [[0.25, 0.75]])


def test_interpolation_inhomogeneous_bc():
    """ test field interpolation with inhomogeneous boundary condition """
    sf = ScalarField(UnitGrid([3, 3], periodic=False))
    x = 1 + np.random.random()
    v = sf.interpolate(
        [x, 0], method="numba", bc=["natural", {"type": "value", "value": "x"}]
    )
    assert x == pytest.approx(v)


@skipUnlessModule("matplotlib")
def test_from_image(tmp_path):
    from matplotlib.pyplot import imsave

    img_data = np.random.uniform(size=(9, 8, 3))
    img_data_gray = img_data @ np.array([0.299, 0.587, 0.114])
    path = tmp_path / "test_from_image.png"
    imsave(path, img_data, vmin=0, vmax=1)
    sf = ScalarField.from_image(path)
    np.testing.assert_allclose(sf.data, img_data_gray.T[:, ::-1], atol=0.05)


def test_to_scalar():
    """ test conversion to scalar field """
    sf = ScalarField.random_uniform(UnitGrid([3, 3]))
    np.testing.assert_allclose(sf.to_scalar().data, sf.data)
    np.testing.assert_allclose(sf.to_scalar("norm_squared").data, sf.data ** 2)
    np.testing.assert_allclose(sf.to_scalar(lambda x: 2 * x).data, 2 * sf.data)

    with pytest.raises(ValueError):
        sf.to_scalar("nonsense")


@pytest.mark.parametrize("method", ["integral", "average"])
def test_projection(example_grid_nd, method):
    """ test scalar projection """
    sf = ScalarField.random_uniform(example_grid_nd)
    for ax in example_grid_nd.axes:
        sp = sf.project(ax, method=method)
        assert sp.grid.dim < example_grid_nd.dim
        assert sp.grid.num_axes == example_grid_nd.num_axes - 1
        if method == "integral":
            assert sp.integral == pytest.approx(sf.integral)
        elif method == "average":
            assert sp.average == pytest.approx(sf.average)

    with pytest.raises(ValueError):
        sf.project(example_grid_nd.axes[0], method="nonsense")


def test_slice(example_grid_nd):
    """ test scalar slicing """
    sf = ScalarField(example_grid_nd, 0.5)
    p = example_grid_nd.get_random_point()
    for i in range(example_grid_nd.num_axes):
        sf_slc = sf.slice({example_grid_nd.axes[i]: p[i]})
        np.testing.assert_allclose(sf_slc.data, 0.5)
        assert sf_slc.grid.dim < example_grid_nd.dim
        assert sf_slc.grid.num_axes == example_grid_nd.num_axes - 1

    with pytest.raises(boundaries.DomainError):
        sf.slice({example_grid_nd.axes[0]: -10})
    with pytest.raises(ValueError):
        sf.slice({"q": 0})


def test_slice_positions():
    """ test scalar slicing at standard positions """
    grid = UnitGrid([3, 1])
    sf = ScalarField(grid, np.arange(3).reshape(3, 1))
    assert sf.slice({"x": "min"}).data == 0
    assert sf.slice({"x": "mid"}).data == 1
    assert sf.slice({"x": "max"}).data == 2

    with pytest.raises(ValueError):
        sf.slice({"x": "foo"})
    with pytest.raises(ValueError):
        sf.slice({"x": 0}, method="nonsense")


def test_interpolation_mutable():
    """ test interpolation on mutable fields """
    grid = UnitGrid([2], periodic=True)
    field = ScalarField(grid)

    for method in ["numba", "scipy"]:
        field.data = 1
        np.testing.assert_allclose(field.interpolate([0.5], method=method), 1)
        field.data = 2
        np.testing.assert_allclose(field.interpolate([0.5], method=method), 2)

    # test overwriting field values
    data = np.full_like(field.data, 3)
    intp = field.make_interpolator(method="numba")
    np.testing.assert_allclose(intp(np.array([0.5]), data), 3)


def test_boundary_interpolation_1d():
    """ test boundary interpolation for 1d fields """
    grid = UnitGrid([5])
    field = ScalarField(grid, np.arange(grid.shape[0]))

    # test boundary interpolation
    bndry_val = 0.25
    for bndry in grid._iter_boundaries():
        val = field.get_boundary_values(*bndry, bc={"value": bndry_val})
        np.testing.assert_allclose(val, bndry_val)

        ev = field.make_get_boundary_values(*bndry, bc={"value": bndry_val})
        out = ev()
        np.testing.assert_allclose(out, bndry_val)
        ev(data=field.data, out=out)
        np.testing.assert_allclose(out, bndry_val)


def test_boundary_interpolation_2d():
    """ test boundary interpolation for 2d fields """
    grid = CartesianGrid([[0.1, 0.3], [-2, 3]], [3, 3])
    field = ScalarField.random_normal(grid)

    # test boundary interpolation
    bndry_val = np.random.randn(3)
    for bndry in grid._iter_boundaries():
        val = field.get_boundary_values(*bndry, bc={"value": bndry_val})
        np.testing.assert_allclose(val, bndry_val)

        ev = field.make_get_boundary_values(*bndry, bc={"value": bndry_val})
        out = ev()
        np.testing.assert_allclose(out, bndry_val)
        ev(data=field.data, out=out)
        np.testing.assert_allclose(out, bndry_val)


def test_numpy_ufuncs():
    """ test numpy ufuncs """
    grid = UnitGrid([2, 2])
    f1 = ScalarField.random_uniform(grid, 0.1, 0.9)

    f2 = np.sin(f1)
    np.testing.assert_allclose(f2.data, np.sin(f1.data))
    f2.data = 0
    np.sin(f1, out=f2)
    np.testing.assert_allclose(f2.data, np.sin(f1.data))

    np.testing.assert_allclose(np.add(f1, 2).data, f1.data + 2)

    with pytest.raises(TypeError):
        np.sum(f1, 1)


def test_plotting_1d():
    """ test plotting of 1d scalar fields """
    grid = UnitGrid([3])
    field = ScalarField.random_uniform(grid, 0.1, 0.9)

    ref = field.plot()
    field._update_plot(ref)


def test_plotting_2d():
    """ test plotting of 2d scalar fields """
    grid = UnitGrid([3, 3])
    field = ScalarField.random_uniform(grid, 0.1, 0.9)

    ref = field.plot()
    field._update_plot(ref)


@skipUnlessModule("napari")
def test_interactive_plotting():
    """ test the interactive plotting """
    grid = UnitGrid([3, 3])
    field = ScalarField.random_uniform(grid, 0.1, 0.9)
    field.plot_interactive(viewer_args={"show": False, "close": True})

    grid = UnitGrid([3, 3, 3])
    field = ScalarField.random_uniform(grid, 0.1, 0.9)
    field.plot_interactive(viewer_args={"show": False, "close": True})


def test_complex_dtype():
    """ test the support of a complex data type """
    grid = UnitGrid([2])
    f = ScalarField(grid, 1j)
    assert f.is_complex
    np.testing.assert_allclose(f.data, np.array([1j, 1j]))

    f = ScalarField(grid, 1)
    assert not f.is_complex
    with pytest.raises(np.core._exceptions.UFuncTypeError):
        f += 1j

    f = f + 1j
    assert f.is_complex
    np.testing.assert_allclose(f.data, np.full((2,), 1 + 1j))
    assert f.integral == pytest.approx(2 + 2j)
    assert f.average == pytest.approx(1 + 1j)
    np.testing.assert_allclose(f.to_scalar("abs").data, np.full((2,), np.sqrt(2)))
    assert f.magnitude == pytest.approx(np.sqrt(2))


def test_complex_plotting():
    """ test plotting of complex fields """
    for dim in (1, 2):
        f = ScalarField(UnitGrid([3] * dim), 1j)
        f.plot()


def test_complex_methods():
    """ test special methods for complex data type """
    grid = UnitGrid([2, 2])
    f = ScalarField(grid, 1j)
    for method in ["scipy", "numba"]:
        val = f.interpolate([1, 1], method=method)
        np.testing.assert_allclose(val, np.array([1j, 1j]))

    f = ScalarField(grid, 1 + 2j)
    np.testing.assert_allclose(f.project("x").data, np.full((2,), 2 + 4j))
    np.testing.assert_allclose(f.slice({"x": 1}).data, np.full((2,), 1 + 2j))


def test_complex_operators():
    """ test differential operators for complex data type """
    f = ScalarField(UnitGrid([2, 2]), 1j)
    assert f.laplace("natural").magnitude == pytest.approx(0)
