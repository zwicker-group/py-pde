"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import CartesianGrid, PolarSymGrid, ScalarField, UnitGrid
from pde.grids.boundaries.local import (
    BCBase,
    BCDataError,
    _get_arr_1d,
    registered_boundary_condition_classes,
    registered_boundary_condition_names,
)


def test_get_arr_1d():
    """test the _get_arr_1d function"""
    # 1d
    a = np.arange(3)
    arr_1d, i, bc_idx = _get_arr_1d(a, [1], 0)
    assert i == 1
    assert bc_idx == (...,)
    np.testing.assert_equal(arr_1d, a)

    # 2d
    a = np.arange(4).reshape(2, 2)
    arr_1d, i, bc_idx = _get_arr_1d(a, [0, 0], 0)
    assert i == 0
    assert bc_idx == (..., 0)
    np.testing.assert_equal(arr_1d, a[:, 0])

    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1], 1)
    assert i == 1
    assert bc_idx == (..., 1)
    np.testing.assert_equal(arr_1d, a[1, :])

    # 3d
    a = np.arange(8).reshape(2, 2, 2)
    arr_1d, i, bc_idx = _get_arr_1d(a, [0, 0, 0], 0)
    assert i == 0
    assert bc_idx == (..., 0, 0)
    np.testing.assert_equal(arr_1d, a[:, 0, 0])

    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1, 0], 1)
    assert i == 1
    assert bc_idx == (..., 1, 0)
    np.testing.assert_equal(arr_1d, a[1, :, 0])

    arr_1d, i, bc_idx = _get_arr_1d(a, [1, 1, 0], 2)
    assert i == 0
    assert bc_idx == (..., 1, 1)
    np.testing.assert_equal(arr_1d, a[1, 1, :])


def test_individual_boundaries():
    """test setting individual boundaries"""
    g = UnitGrid([2])
    for data in [
        "value",
        {"value": 1},
        {"type": "value", "value": 1},
        "derivative",
        {"derivative": 1},
        {"type": "derivative", "value": 1},
        {"mixed": 1},
        {"type": "mixed", "value": 1},
        "extrapolate",
    ]:
        bc = BCBase.from_data(g, 0, upper=True, data=data, rank=0)

        assert isinstance(str(bc), str)
        assert isinstance(repr(bc), str)
        assert "field" in bc.get_mathematical_representation("field")
        assert bc.rank == 0
        assert bc.homogeneous
        bc.check_value_rank(0)
        with pytest.raises(RuntimeError):
            bc.check_value_rank(1)

        for bc_copy in [BCBase.from_data(g, 0, upper=True, data=bc, rank=0), bc.copy()]:
            assert bc == bc_copy


def test_individual_boundaries_multidimensional():
    """test setting individual boundaries in 2d"""
    g2 = UnitGrid([2, 3])
    bc = BCBase.from_data(g2, 0, True, {"type": "value", "value": [1, 2]}, rank=1)

    assert isinstance(str(bc), str)
    assert isinstance(repr(bc), str)
    assert "field" in bc.get_mathematical_representation("field")
    assert bc.rank == 1
    assert bc.homogeneous
    assert bc.axis_coord == 2
    bc.check_value_rank(1)
    with pytest.raises(RuntimeError):
        bc.check_value_rank(0)

    for bc_copy in [BCBase.from_data(g2, 0, upper=True, data=bc, rank=1), bc.copy()]:
        assert bc == bc_copy


def test_virtual_points():
    """test the calculation of virtual points"""
    g = UnitGrid([2])
    data = np.array([1, 2])

    # test constant boundary conditions
    bc = BCBase.from_data(g, 0, False, {"type": "value", "value": 1})
    assert bc.get_virtual_point(data) == pytest.approx(1)
    assert not bc.value_is_linked
    bc = BCBase.from_data(g, 0, True, {"type": "value", "value": 1})
    assert bc.get_virtual_point(data) == pytest.approx(0)
    assert not bc.value_is_linked

    # test derivative boundary conditions (wrt to outwards derivative)
    for up, b, val in [
        (False, {"type": "derivative", "value": -1}, 0),
        (True, {"type": "derivative", "value": 1}, 3),
        (False, "extrapolate", 0),
        (True, "extrapolate", 3),
        (False, {"type": "mixed", "value": 4, "const": 1}, 0),
        (True, {"type": "mixed", "value": 2, "const": 4}, 2),
    ]:
        bc = BCBase.from_data(g, 0, up, b)
        assert bc.upper == up
        assert bc.get_virtual_point(data) == pytest.approx(val)
        assert not bc.value_is_linked
        ev = bc.make_virtual_point_evaluator()
        assert ev(data, (2,) if up else (-1,)) == pytest.approx(val)

    # test curvature for y = 4 * x**2
    data = np.array([1, 9])
    bc = BCBase.from_data(g, 0, False, {"type": "curvature", "value": 8})
    assert bc.get_virtual_point(data) == pytest.approx(1)
    assert not bc.value_is_linked
    bc = BCBase.from_data(g, 0, True, {"type": "curvature", "value": 8})
    assert bc.get_virtual_point(data) == pytest.approx(25)
    assert not bc.value_is_linked


@pytest.mark.parametrize("upper", [False, True])
def test_virtual_points_linked_data(upper):
    """test the calculation of virtual points with linked_data"""
    g = UnitGrid([2, 2])
    point = (1, 1) if upper else (0, 0)
    data = np.zeros(g.shape)

    # test constant boundary conditions
    bc_data = np.array([1, 1])
    bc = BCBase.from_data(g, 0, upper, {"type": "value", "value": bc_data})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 3

    assert bc.get_virtual_point(data, point) == pytest.approx(6)
    ev = bc.make_virtual_point_evaluator()
    assert ev(data, point) == pytest.approx(6)

    # test derivative boundary conditions (wrt to outwards derivative)
    bc = BCBase.from_data(g, 0, upper, {"type": "derivative", "value": bc_data})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 4

    assert bc.get_virtual_point(data, point) == pytest.approx(4)
    ev = bc.make_virtual_point_evaluator()
    assert ev(data, point) == pytest.approx(4)

    # test derivative boundary conditions (wrt to outwards derivative)
    bc = BCBase.from_data(g, 0, upper, {"type": "mixed", "value": bc_data, "const": 3})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 4

    assert bc.get_virtual_point(data, point) == pytest.approx(1)
    ev = bc.make_virtual_point_evaluator()
    assert ev(data, point) == pytest.approx(1)


def test_mixed_condition():
    """test the calculation of virtual points"""
    g = UnitGrid([2])
    data = np.array([1, 2])

    for upper in [True, False]:
        bc1 = BCBase.from_data(g, 0, upper, {"type": "mixed", "value": 0, "const": 2})
        bc2 = BCBase.from_data(g, 0, upper, {"derivative": 2})
        assert bc1.get_virtual_point(data) == pytest.approx(bc2.get_virtual_point(data))

    bc = BCBase.from_data(g, 0, False, {"type": "mixed", "value": np.inf})
    assert bc.get_virtual_point(data) == pytest.approx(-1)
    bc = BCBase.from_data(g, 0, True, {"type": "mixed", "value": np.inf})
    assert bc.get_virtual_point(data) == pytest.approx(-2)

    g = UnitGrid([2, 2])
    data = np.ones(g.shape)
    bc = BCBase.from_data(
        g, 0, False, {"type": "mixed", "value": [1, 2], "const": [3, 4]}
    )
    assert bc.get_virtual_point(data, (0, 0)) == pytest.approx(2 + 1 / 3)
    assert bc.get_virtual_point(data, (0, 1)) == pytest.approx(2)


def test_inhomogeneous_bcs_1d():
    """test inhomogeneous boundary conditions in 1d grids"""
    g = UnitGrid([2])
    data = np.ones((2,))  # field is 1 everywhere

    # first order bc
    bc_x = BCBase.from_data(g, 0, True, {"value": "x**2"})
    assert isinstance(str(bc_x), str)
    assert bc_x.rank == 0
    assert bc_x.axis_coord == 2
    assert bc_x.get_virtual_point(data, (1,)) == pytest.approx(7.0)
    ev = bc_x.make_virtual_point_evaluator()
    assert ev(data, (1,)) == pytest.approx(7.0)

    # test lower bc
    bc_x = BCBase.from_data(g, 0, False, {"value": "x**2"})
    assert bc_x.rank == 0
    assert bc_x.axis_coord == 0
    assert bc_x.get_virtual_point(data, (0,)) == pytest.approx(-1.0)
    ev = bc_x.make_virtual_point_evaluator()
    assert ev(data, (0,)) == pytest.approx(-1.0)


def test_inhomogeneous_bcs_2d():
    """test inhomogeneous boundary conditions in 2d grids"""
    g = UnitGrid([2, 2])
    data = np.ones((2, 2))

    # first order bc
    bc_x = BCBase.from_data(g, 0, True, {"value": "y"})
    assert isinstance(str(bc_x), str)
    assert bc_x.rank == 0
    assert bc_x.axis_coord == 2
    assert bc_x.get_virtual_point(data, (1, 0)) == pytest.approx(0)
    assert bc_x.get_virtual_point(data, (1, 1)) == pytest.approx(2)

    # second order bc
    bc_x = BCBase.from_data(g, 0, True, {"curvature": "y"})
    assert isinstance(str(bc_x), str)
    assert bc_x.rank == 0
    assert bc_x.axis_coord == 2
    assert bc_x.get_virtual_point(data, (1, 0)) == pytest.approx(1.5)
    assert bc_x.get_virtual_point(data, (1, 1)) == pytest.approx(2.5)

    ev = bc_x.make_virtual_point_evaluator()
    assert ev(data, (1, 0)) == pytest.approx(1.5)
    assert ev(data, (1, 1)) == pytest.approx(2.5)

    ev = bc_x.make_adjacent_evaluator()
    assert ev(*_get_arr_1d(data, (0, 0), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (0, 1), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (1, 0), axis=0)) == pytest.approx(1.5)
    assert ev(*_get_arr_1d(data, (1, 1), axis=0)) == pytest.approx(2.5)

    # test lower bc
    bc_x = BCBase.from_data(g, 0, False, {"curvature": "y"})
    assert bc_x.axis_coord == 0
    ev = bc_x.make_adjacent_evaluator()
    assert ev(*_get_arr_1d(data, (1, 0), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (1, 1), axis=0)) == pytest.approx(1)
    assert ev(*_get_arr_1d(data, (0, 0), axis=0)) == pytest.approx(1.5)
    assert ev(*_get_arr_1d(data, (0, 1), axis=0)) == pytest.approx(2.5)


@pytest.mark.parametrize("expr", ["1", "x + y**2"])
def test_expression_bc_setting(expr):
    """test boundary conditions that use an expression"""
    grid = CartesianGrid([[0, 1], [0, 1]], 4)
    bc1 = grid.get_boundary_conditions({"value": expr})
    bc2 = grid.get_boundary_conditions({"virtual_point": f"2 * ({expr}) - value"})

    field = ScalarField.random_uniform(grid)

    f1 = field.copy()
    f1.set_ghost_cells(bc1)
    f2 = field.copy()
    f2.set_ghost_cells(bc2)

    f3 = field.copy()
    bc1.make_ghost_cell_setter()(f3._data_full)
    f4 = field.copy()
    bc2.make_ghost_cell_setter()(f4._data_full)

    np.testing.assert_almost_equal(f1._data_full, f2._data_full)
    np.testing.assert_almost_equal(f1._data_full, f3._data_full)
    np.testing.assert_almost_equal(f1._data_full, f4._data_full)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_expression_bc_operator(dim):
    """test boundary conditions that use an expression in an operator"""
    grid = CartesianGrid([[0, 1]] * dim, 4)
    bc1 = grid.get_boundary_conditions({"value": 1})
    bc2 = grid.get_boundary_conditions({"virtual_point": f"2 - value"})
    assert "field" in bc2.get_mathematical_representation("field")

    field = ScalarField(grid, 1)

    result = field.laplace(bc1)
    np.testing.assert_allclose(result.data, 0.0)
    result = field.laplace(bc2)
    np.testing.assert_allclose(result.data, 0.0)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_expression_bc_value(dim):
    """test boundary conditions that use an expression to calculate the value"""
    grid = CartesianGrid([[0, 1]] * dim, 4)
    bc1 = grid.get_boundary_conditions({"value": 1})
    bc2 = grid.get_boundary_conditions({"value_expression": "1"})
    assert "field" in bc2.get_mathematical_representation("field")

    field = ScalarField(grid, 1)

    result = field.laplace(bc1)
    np.testing.assert_allclose(result.data, 0.0)
    result = field.laplace(bc2)
    np.testing.assert_allclose(result.data, 0.0)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_expression_bc_derivative(dim):
    """test boundary conditions that use an expression to calculate the derivative"""
    grid = CartesianGrid([[0, 1]] * dim, 4)
    bc1 = grid.get_boundary_conditions({"derivative": 0})
    bc2 = grid.get_boundary_conditions({"derivative_expression": "0"})
    assert "field" in bc2.get_mathematical_representation("field")

    field = ScalarField(grid, 1)

    result = field.laplace(bc1)
    np.testing.assert_allclose(result.data, 0.0)
    result = field.laplace(bc2)
    np.testing.assert_allclose(result.data, 0.0)


def test_expression_invalid_args():
    """test boundary conditions use an expression with invalid data"""
    grid = CartesianGrid([[0, 1]], 4)
    with pytest.raises(BCDataError):
        grid.get_boundary_conditions({"derivative_expression": "heaviside(x)"})


def test_expression_bc_polar_grid():
    """test whether expression BCs work on polar grids"""
    grid = PolarSymGrid(radius=1, shape=8)
    bcs = grid.get_boundary_conditions([{"value": 1}, {"value_expression": "1"}])

    state = ScalarField.from_expression(grid, "0")
    bcs.set_ghost_cells(state._data_full)
    np.testing.assert_allclose(state.data, 0)
    assert state._data_full[0] == state._data_full[-1] == 2

    state._data_full[...] = 0
    bc_setter = bcs.make_ghost_cell_setter()
    bc_setter(state._data_full)
    np.testing.assert_allclose(state.data, 0)
    assert state._data_full[0] == state._data_full[-1] == 2


def test_getting_registered_bcs():
    """test the functions that return the registered BCs"""
    assert isinstance(registered_boundary_condition_classes(), dict)
    assert isinstance(registered_boundary_condition_names(), dict)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("target", ["value", "derivative"])
def test_user_bcs_numpy(dim, target):
    """test setting user BCs"""
    value = np.arange(3) if dim == 2 else 1
    grid = UnitGrid([3] * dim)
    bcs = grid.get_boundary_conditions({"type": "user"})

    # use normal method to set BCs
    f1 = ScalarField(grid)
    f1.set_ghost_cells(bc={target: value})

    # use user method to set BCs
    f2 = ScalarField(grid)

    # test whether normal call is a no-op
    f2._data_full = 3
    f2.set_ghost_cells(bc=bcs)
    np.testing.assert_allclose(f2._data_full, 3)
    f2.set_ghost_cells(bc=bcs, args={"t": 1})
    np.testing.assert_allclose(f2._data_full, 3)
    f2._data_full = 0

    # test whether calling setter with user data works properly
    bcs.set_ghost_cells(f2._data_full, args={target: value})

    np.testing.assert_allclose(f1._data_full, f2._data_full)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("target", ["value", "derivative"])
def test_user_bcs_numba(dim, target):
    """test setting user BCs"""
    if dim == 1:
        value = 1
    elif dim == 2:
        value = np.arange(3)
    elif dim == 3:
        value = np.arange(9).reshape(3, 3)
    grid = UnitGrid([3] * dim)
    bcs = grid.get_boundary_conditions({"type": "user"})

    # use normal method to set BCs
    f1 = ScalarField(grid)
    f1.set_ghost_cells(bc={target: value})

    # use user method to set BCs
    f2 = ScalarField(grid)
    setter = bcs.make_ghost_cell_setter()

    # test whether normal call is a no-op
    f2._data_full = 3
    setter(f2._data_full)
    np.testing.assert_allclose(f2._data_full, 3)
    setter(f2._data_full, args={"t": 1})
    np.testing.assert_allclose(f2._data_full, 3)
    f2._data_full = 0

    # test whether calling setter with user data works properly
    setter(f2._data_full, args={target: value})
    np.testing.assert_allclose(f1._data_full, f2._data_full)
