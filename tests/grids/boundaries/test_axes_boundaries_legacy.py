"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import itertools

import numpy as np
import pytest

from pde import ScalarField, UnitGrid, config
from pde.grids.base import PeriodicityError
from pde.grids.boundaries.axes import (
    BCDataError,
    BoundariesBase,
    BoundariesList,
    BoundariesSetter,
)
from pde.grids.boundaries.axis import BoundaryPair, BoundaryPeriodic, get_boundary_axis
from pde.grids.boundaries.local import NeumannBC


def test_boundaries_legacy():
    """Test setting boundaries for multiple systems."""
    with config({"boundaries.accept_lists": True}):
        b = ["periodic", "value", {"type": "derivative", "value": 1}]
        for bx, by in itertools.product(b, b):
            periodic = [b == "periodic" for b in (bx, by)]
            g = UnitGrid([2, 2], periodic=periodic)

            bcs = BoundariesBase.from_data([bx, by], grid=g)
            bc_x = get_boundary_axis(g, 0, bx)
            bc_y = get_boundary_axis(g, 1, by)

            assert bcs.grid.num_axes == 2
            assert bcs.periodic == periodic
            assert bcs[0] == bc_x
            assert bcs[1] == bc_y
            assert "field" in bcs.get_mathematical_representation("field")
            assert isinstance(str(bcs), str)
            assert isinstance(repr(bcs), str)

            assert bcs == BoundariesBase.from_data([bc_x, bc_y], grid=g)
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


def test_boundary_specifications_legacy():
    """Test different ways of specifying boundary conditions."""
    with config({"boundaries.accept_lists": True}):
        g = UnitGrid([2])
        bc1 = BoundariesBase.from_data(
            [{"type": "derivative", "value": 0}, {"type": "value", "value": 0}], grid=g
        )
        assert bc1 == BoundariesBase.from_data(
            [{"type": "derivative"}, {"type": "value"}], grid=g
        )
        assert bc1 == BoundariesBase.from_data(
            [{"derivative": 0}, {"value": 0}], grid=g
        )
        assert bc1 == BoundariesBase.from_data(["neumann", "dirichlet"], grid=g)


def test_boundaries_property_legacy():
    """Test boundaries property."""
    with config({"boundaries.accept_lists": True}):
        g = UnitGrid([2, 2])
        bc = BoundariesBase.from_data(["neumann", "dirichlet"], grid=g)
        assert len(list(bc.boundaries)) == 4

        bc = BoundariesBase.from_data("neumann", grid=g)
        for b in bc.boundaries:
            assert isinstance(b, NeumannBC)

        g = UnitGrid([2, 2], periodic=[True, False])
        bc = BoundariesBase.from_data("auto_periodic_neumann", grid=g)
        assert len(list(bc.boundaries)) == 2


def test_boundary_specifications_legacy_disabled():
    """Test disabling legacy boundary conditions."""
    with config({"boundaries.accept_lists": False}):
        g = UnitGrid([2])
        with pytest.raises(BCDataError):
            bc1 = BoundariesBase.from_data(
                [{"type": "derivative", "value": 0}, {"type": "value", "value": 0}],
                grid=g,
            )
        with pytest.raises(BCDataError):
            BoundariesBase.from_data(
                [{"type": "derivative"}, {"type": "value"}], grid=g
            )
        with pytest.raises(BCDataError):
            BoundariesBase.from_data([{"derivative": 0}, {"value": 0}], grid=g)
        with pytest.raises(BCDataError):
            BoundariesBase.from_data(["neumann", "dirichlet"], grid=g)
