"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools

import numpy as np
import pytest
from pde import ScalarField, UnitGrid
from pde.grids.boundaries.axes import Boundaries
from pde.grids.boundaries.axis import BoundaryPair, BoundaryPeriodic, get_boundary_axis


def test_boundaries():
    """ test setting boundaries for multiple systems """
    b = ["periodic", "value", {"type": "derivative", "value": 1}]
    for bx, by in itertools.product(b, b):
        g = UnitGrid([2, 2], periodic=[b == "periodic" for b in (bx, by)])

        bcs = Boundaries.from_data(g, [bx, by])
        bc_x = get_boundary_axis(g, 0, bx)
        bc_y = get_boundary_axis(g, 1, by)

        assert bcs.grid.num_axes == 2
        assert bcs[0] == bc_x
        assert bcs[1] == bc_y
        assert bcs == Boundaries.from_data(g, [bc_x, bc_y])
        if bx == by:
            assert bcs == Boundaries.from_data(g, bx)

        bc2 = bcs.copy()
        assert bcs == bc2
        assert bcs is not bc2

    b1 = Boundaries.from_data(UnitGrid([2, 2]), "natural")
    b2 = Boundaries.from_data(UnitGrid([3, 3]), "natural")
    assert b1 != b2


def test_boundary_specifications():
    """ test different ways of specifying boundary conditions """
    g = UnitGrid([2])
    bc1 = Boundaries.from_data(
        g, [{"type": "derivative", "value": 0}, {"type": "value", "value": 0}]
    )
    assert bc1 == Boundaries.from_data(g, [{"type": "derivative"}, {"type": "value"}])
    assert bc1 == Boundaries.from_data(g, [{"derivative": 0}, {"value": 0}])
    assert bc1 == Boundaries.from_data(g, ["neumann", "dirichlet"])


def test_mixed_boundary_condition():
    """ test limiting cases of the mixed boundary condition """
    g = UnitGrid([2])
    d = np.random.random(2)
    g1 = g.get_operator("gradient", bc=[{"mixed": 0}, {"mixed": np.inf}])
    g2 = g.get_operator("gradient", bc=["derivative", "value"])
    np.testing.assert_allclose(g1(d), g2(d))


@pytest.mark.parametrize(
    "cond,is_value",
    [
        ("natural", False),
        ("auto_periodic_neumann", False),
        ("auto_periodic_dirichlet", True),
    ],
)
def test_natural_boundary_conditions(cond, is_value):
    """ test special automatic boundary conditions """
    g = UnitGrid([2, 2], periodic=[True, False])
    for bc in [
        Boundaries.from_data(g, cond),
        Boundaries.from_data(g, ["periodic", cond]),
    ]:
        assert isinstance(bc[0], BoundaryPeriodic)
        if is_value:
            assert bc[1] == BoundaryPair.from_data(g, 1, "value")
        else:
            assert bc[1] == BoundaryPair.from_data(g, 1, "derivative")


def test_special_cases():
    """ test some special boundary conditions """
    g = UnitGrid([5])
    s = ScalarField(g, np.arange(5))
    for bc in ["extrapolate", {"curvature": 0}]:
        np.testing.assert_allclose(s.laplace(bc).data, 0)


def test_bc_values():
    """ test setting the values of boundary conditions """
    g = UnitGrid([5])
    bc = g.get_boundary_conditions([{"value": 2}, {"derivative": 3}])
    assert bc[0].low.value == 2 and bc[0].high.value == 3
    bc.scale_value(5)
    assert bc[0].low.value == 10 and bc[0].high.value == 15
    bc.set_value(7)
    assert bc[0].low.value == bc[0].high.value == 7
