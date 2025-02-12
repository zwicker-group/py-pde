"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools

import numpy as np
import pytest

from pde import PolarSymGrid, ScalarField, UnitGrid
from pde.grids.base import PeriodicityError
from pde.grids.boundaries.axes import (
    BCDataError,
    BoundariesBase,
    BoundariesList,
    BoundariesSetter,
)
from pde.grids.boundaries.axis import BoundaryPair, BoundaryPeriodic, get_boundary_axis
from pde.grids.boundaries.local import NeumannBC


def test_boundaries():
    """Test setting boundaries for multiple systems."""
    b = ["periodic", "value", {"type": "derivative", "value": 1}]
    for bx, by in itertools.product(b, b):
        periodic = [b == "periodic" for b in (bx, by)]
        g = UnitGrid([2, 2], periodic=periodic)

        bcs = BoundariesBase.from_data({"x": bx, "y": by}, grid=g)
        bc_x = get_boundary_axis(g, 0, bx)
        bc_y = get_boundary_axis(g, 1, by)

        assert bcs.grid.num_axes == 2
        assert bcs.periodic == periodic
        assert bcs[0] == bc_x
        assert bcs[1] == bc_y
        assert "field" in bcs.get_mathematical_representation("field")
        assert isinstance(str(bcs), str)
        assert isinstance(repr(bcs), str)

        assert bcs == BoundariesBase.from_data({"x": bc_x, "y": bc_y}, grid=g)
        if bx == by:
            assert bcs == BoundariesBase.from_data(bx, grid=g)

        bc2 = bcs.copy()
        assert bcs == bc2
        assert bcs is not bc2

    b1 = BoundariesBase.from_data("auto_periodic_neumann", grid=UnitGrid([2, 2]))
    b2 = BoundariesBase.from_data("auto_periodic_neumann", grid=UnitGrid([3, 3]))
    assert isinstance(b1, BoundariesList)
    assert isinstance(b2, BoundariesList)
    assert b1 != b2


def test_boundaries_edge_cases():
    """Test treatment of invalid data."""
    grid = UnitGrid([3, 3])
    bcs = grid.get_boundary_conditions("auto_periodic_neumann")
    with pytest.raises(BCDataError):
        BoundariesList([])
    with pytest.raises(BCDataError):
        BoundariesList([bcs[0]])
    with pytest.raises(BCDataError):
        BoundariesList([bcs[0], bcs[0]])

    assert bcs == BoundariesList([bcs[0], bcs[1]])
    bc0 = get_boundary_axis(grid.copy(), 0, "auto_periodic_neumann")
    assert bcs == BoundariesList([bc0, bcs[1]])
    bc0 = get_boundary_axis(UnitGrid([4, 3]), 0, "auto_periodic_neumann")
    with pytest.raises(BCDataError):
        BoundariesList([bc0, bcs[1]])
    bc0 = get_boundary_axis(UnitGrid([3, 3], periodic=True), 0, "auto_periodic_neumann")
    with pytest.raises(BCDataError):
        BoundariesList([bc0, bcs[1]])


def test_boundary_specifications():
    """Test different ways of specifying boundary conditions."""
    g = UnitGrid([2])
    ref = BoundariesBase.from_data({"x-": "neumann", "x+": "dirichlet"}, grid=g)
    BC_DATA = [
        {"x-": {"type": "derivative", "value": 0}, "x+": {"type": "value", "value": 0}},
        {"left": {"type": "derivative"}, "right": {"type": "value"}},
        {"x-": {"derivative": 0}, "x+": {"value": 0}},
    ]
    for data in BC_DATA:
        assert ref == BoundariesBase.from_data(data, grid=g)


def test_mixed_boundary_condition(rng):
    """Test limiting cases of the mixed boundary condition."""
    g = UnitGrid([2])
    d = rng.random(2)
    g1 = g.make_operator("gradient", bc={"x-": {"mixed": 0}, "x+": {"mixed": np.inf}})
    g2 = g.make_operator("gradient", bc={"x-": "derivative", "x+": "value"})
    np.testing.assert_allclose(g1(d), g2(d))


@pytest.mark.parametrize(
    "cond,is_value",
    [
        ("auto_periodic_neumann", False),
        ("auto_periodic_dirichlet", True),
    ],
)
def test_natural_boundary_conditions(cond, is_value):
    """Test special automatic boundary conditions."""
    g = UnitGrid([2, 2], periodic=[True, False])
    for bc in [
        BoundariesBase.from_data(cond, grid=g),
        BoundariesBase.from_data({"x": "periodic", "y": cond}, grid=g),
    ]:
        assert isinstance(bc[0], BoundaryPeriodic)
        if is_value:
            assert bc[1] == BoundaryPair.from_data(g, 1, "value")
        else:
            assert bc[1] == BoundaryPair.from_data(g, 1, "derivative")


def test_special_cases():
    """Test some special boundary conditions."""
    g = UnitGrid([5])
    s = ScalarField(g, np.arange(5))
    for bc in ["extrapolate", {"curvature": 0}]:
        np.testing.assert_allclose(s.laplace(bc).data, 0)


def test_bc_values():
    """Test setting the values of boundary conditions."""
    g = UnitGrid([5])
    bc = g.get_boundary_conditions({"x-": {"value": 2}, "x+": {"derivative": 3}})
    assert bc[0].low.value == 2
    assert bc[0].high.value == 3


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_set_ghost_cells(dim, periodic, rng):
    """Test setting values for ghost cells."""
    grid = UnitGrid([1] * dim, periodic=periodic)
    field = ScalarField.random_uniform(grid, rng=rng)
    bcs = grid.get_boundary_conditions("auto_periodic_neumann")

    arr1 = field._data_full.copy()
    bcs.set_ghost_cells(arr1)

    arr2 = field._data_full.copy()
    setter = bcs.make_ghost_cell_setter()
    setter(arr2)

    # test valid BCs:
    for n in range(dim):
        idx = [slice(1, -1)] * dim
        idx[n] = slice(None)
        np.testing.assert_allclose(arr1[tuple(idx)], arr2[tuple(idx)])


def test_setting_specific_bcs():
    """Test the interface of setting specific conditions."""
    grid = UnitGrid([4, 4], periodic=[False, True])
    bcs = grid.get_boundary_conditions("auto_periodic_neumann")

    # test non-periodic axis
    assert str(bcs[0]) == '"derivative"'
    assert str(bcs["x"]) == '"derivative"'
    bcs["x"] = "value"
    assert str(bcs["x"]) == '"value"'
    bcs["left"] = "derivative"
    assert str(bcs["left"]) == '"derivative"'
    assert str(bcs["x-"]) == '"derivative"'
    assert str(bcs["right"]) == '"value"'
    assert str(bcs["x+"]) == '"value"'
    bcs["right"] = "derivative"
    assert str(bcs["x"]) == '"derivative"'
    bcs["x-"] = bcs["x+"] = "value"
    assert str(bcs["x"]) == '"value"'
    with pytest.raises(PeriodicityError):
        bcs["x"] = "periodic"

    # test periodic axis
    assert str(bcs[1]) == '"periodic"'
    assert str(bcs["y"]) == '"periodic"'
    assert str(bcs["top"]) == '"periodic"'
    bcs["y"] = "periodic"
    with pytest.raises(PeriodicityError):
        bcs["y"] = "value"
    with pytest.raises(PeriodicityError):
        bcs["top"] = "value"
    with pytest.raises(PeriodicityError):
        bcs["y+"] = "value"

    # test wrong input
    with pytest.raises(KeyError):
        bcs["nonsense"]
    with pytest.raises(TypeError):
        bcs[None]
    with pytest.raises(KeyError):
        bcs["nonsense"] = None


def test_boundaries_property():
    """Test boundaries property."""
    g = UnitGrid([2, 2])
    bc = BoundariesBase.from_data({"x": "neumann", "y": "dirichlet"}, grid=g)
    assert len(list(bc.boundaries)) == 4

    bc = BoundariesBase.from_data("neumann", grid=g)
    for b in bc.boundaries:
        assert isinstance(b, NeumannBC)

    g = UnitGrid([2, 2], periodic=[True, False])
    bc = BoundariesBase.from_data("auto_periodic_neumann", grid=g)
    assert len(list(bc.boundaries)) == 2


@pytest.mark.parametrize("periodic", [True, False])
def test_boundaries_setter_1d(periodic, rng):
    """Test BoundariesSetter class for 1d grids."""

    def setter(data, args=None):
        if periodic:
            data[0] = data[-2]
            data[-1] = data[1]
        else:
            data[0] = data[1]  # Neumann
            data[-1] = -data[-2]  # Dirichlet

    f1 = ScalarField.random_normal(UnitGrid([4], periodic=periodic))
    f2 = f1.copy()

    f1.set_ghost_cells(bc=BoundariesSetter(setter))
    if periodic:
        f2.set_ghost_cells(bc="periodic")
    else:
        f2.set_ghost_cells(bc={"x-": "neumann", "x+": "dirichlet"})
    np.testing.assert_allclose(f1._data_full, f2._data_full)


def test_boundaries_setter_2d(rng):
    """Test BoundariesSetter class for 2d grids."""

    def setter(data, args=None):
        data[0, :] = data[1, :]  # Neumann
        data[-1, :] = -data[-2, :]  # Dirichlet
        data[:, 0] = data[:, -2]  # periodic
        data[:, -1] = data[:, 1]  # periodic

    f1 = ScalarField.random_normal(UnitGrid([4, 4], periodic=[False, True]))
    f2 = f1.copy()

    f1.set_ghost_cells(bc=BoundariesSetter(setter))
    f2.set_ghost_cells(bc={"x-": "neumann", "x+": "dirichlet", "y": "periodic"})
    # compare full fields without corner points
    mask = np.ones((6, 6), dtype=bool)
    mask[0, 0] = mask[-1, 0] = mask[0, -1] = mask[-1, -1] = False
    np.testing.assert_allclose(f1._data_full[mask], f2._data_full[mask])


def test_boundaries_axis_synonyms():
    """Test whether we can use synonyms of the axis to set boundaries."""
    grid = PolarSymGrid([1, 2], 3)
    expect = grid.get_boundary_conditions({"r": "value"})

    gbc = grid.get_boundary_conditions
    assert expect == gbc({"radius": "value"})
    assert expect == gbc({"r-": "value", "r+": "value"})
    assert expect == gbc({"radius-": "value", "radius+": "value"})
    assert expect == gbc({"r-": "value", "radius+": "value"})
    assert expect == gbc({"radius-": "value", "r+": "value"})

    with pytest.raises(KeyError):
        gbc({"radius": "value", "r": "value"})
    with pytest.raises(KeyError):
        gbc({"r": "value", "radius": "value"})
    with pytest.raises(KeyError):
        gbc({"radius+": "value", "r+": "value"})
    with pytest.raises(KeyError):
        gbc({"r-": "value", "radius-": "value"})
