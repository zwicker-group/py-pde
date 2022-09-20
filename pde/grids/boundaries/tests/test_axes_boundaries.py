"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools

import numpy as np
import pytest

from pde import ScalarField, UnitGrid
from pde.grids.base import PeriodicityError
from pde.grids.boundaries.axes import BCDataError, Boundaries
from pde.grids.boundaries.axis import BoundaryPair, BoundaryPeriodic, get_boundary_axis


def test_boundaries():
    """test setting boundaries for multiple systems"""
    b = ["periodic", "value", {"type": "derivative", "value": 1}]
    for bx, by in itertools.product(b, b):
        periodic = [b == "periodic" for b in (bx, by)]
        g = UnitGrid([2, 2], periodic=periodic)

        bcs = Boundaries.from_data(g, [bx, by])
        bc_x = get_boundary_axis(g, 0, bx)
        bc_y = get_boundary_axis(g, 1, by)

        assert bcs.grid.num_axes == 2
        assert bcs.periodic == periodic
        assert bcs[0] == bc_x
        assert bcs[1] == bc_y
        assert "field" in bcs.get_mathematical_representation("field")
        assert isinstance(str(bcs), str)
        assert isinstance(repr(bcs), str)

        assert bcs == Boundaries.from_data(g, [bc_x, bc_y])
        if bx == by:
            assert bcs == Boundaries.from_data(g, bx)

        bc2 = bcs.copy()
        assert bcs == bc2
        assert bcs is not bc2

    b1 = Boundaries.from_data(UnitGrid([2, 2]), "auto_periodic_neumann")
    b2 = Boundaries.from_data(UnitGrid([3, 3]), "auto_periodic_neumann")
    assert b1 != b2


def test_boundaries_edge_cases():
    """test treatment of invalid data"""
    grid = UnitGrid([3, 3])
    bcs = grid.get_boundary_conditions("auto_periodic_neumann")
    with pytest.raises(BCDataError):
        Boundaries([])
    with pytest.raises(BCDataError):
        Boundaries([bcs[0]])
    with pytest.raises(BCDataError):
        Boundaries([bcs[0], bcs[0]])

    assert bcs == Boundaries([bcs[0], bcs[1]])
    bc0 = get_boundary_axis(grid.copy(), 0, "auto_periodic_neumann")
    assert bcs == Boundaries([bc0, bcs[1]])
    bc0 = get_boundary_axis(UnitGrid([4, 3]), 0, "auto_periodic_neumann")
    with pytest.raises(BCDataError):
        Boundaries([bc0, bcs[1]])
    bc0 = get_boundary_axis(UnitGrid([3, 3], periodic=True), 0, "auto_periodic_neumann")
    with pytest.raises(BCDataError):
        Boundaries([bc0, bcs[1]])


def test_boundary_specifications():
    """test different ways of specifying boundary conditions"""
    g = UnitGrid([2])
    bc1 = Boundaries.from_data(
        g, [{"type": "derivative", "value": 0}, {"type": "value", "value": 0}]
    )
    assert bc1 == Boundaries.from_data(g, [{"type": "derivative"}, {"type": "value"}])
    assert bc1 == Boundaries.from_data(g, [{"derivative": 0}, {"value": 0}])
    assert bc1 == Boundaries.from_data(g, ["neumann", "dirichlet"])


def test_mixed_boundary_condition():
    """test limiting cases of the mixed boundary condition"""
    g = UnitGrid([2])
    d = np.random.random(2)
    g1 = g.make_operator("gradient", bc=[{"mixed": 0}, {"mixed": np.inf}])
    g2 = g.make_operator("gradient", bc=["derivative", "value"])
    np.testing.assert_allclose(g1(d), g2(d))


@pytest.mark.parametrize(
    "cond,is_value",
    [
        ("auto_periodic_neumann", False),
        ("auto_periodic_dirichlet", True),
    ],
)
def test_natural_boundary_conditions(cond, is_value):
    """test special automatic boundary conditions"""
    g = UnitGrid([2, 2], periodic=[True, False])
    for bc in [
        Boundaries.from_data(g, cond),
        Boundaries.from_data(g, ["periodic", cond]),
    ]:
        assert isinstance(bc[0], BoundaryPeriodic)
        print(bc[1])
        if is_value:
            assert bc[1] == BoundaryPair.from_data(g, 1, "value")
        else:
            assert bc[1] == BoundaryPair.from_data(g, 1, "derivative")


def test_special_cases():
    """test some special boundary conditions"""
    g = UnitGrid([5])
    s = ScalarField(g, np.arange(5))
    for bc in ["extrapolate", {"curvature": 0}]:
        np.testing.assert_allclose(s.laplace(bc).data, 0)


def test_bc_values():
    """test setting the values of boundary conditions"""
    g = UnitGrid([5])
    bc = g.get_boundary_conditions([{"value": 2}, {"derivative": 3}])
    assert bc[0].low.value == 2 and bc[0].high.value == 3


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("periodic", [True, False])
def test_set_ghost_cells(dim, periodic):
    """test setting values for ghost cells"""
    grid = UnitGrid([1] * dim, periodic=periodic)
    field = ScalarField.random_uniform(grid)
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
    """test the interface of setting specific conditions"""
    grid = UnitGrid([4, 4], periodic=[False, True])
    bcs = grid.get_boundary_conditions("auto_periodic_neumann")

    # test non-periodic axis
    assert str(bcs[0]) == '"derivative"'
    assert str(bcs["x"]) == '"derivative"'
    bcs["x"] = "value"
    assert str(bcs["x"]) == '"value"'
    bcs["left"] = "derivative"
    assert str(bcs["left"]) == '"derivative"'
    assert str(bcs["right"]) == '"value"'
    bcs["right"] = "derivative"
    assert str(bcs["x"]) == '"derivative"'
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

    # test wrong input
    with pytest.raises(KeyError):
        bcs["nonsense"]
    with pytest.raises(TypeError):
        bcs[None]
    with pytest.raises(KeyError):
        bcs["nonsense"] = None

    # test different ranks
