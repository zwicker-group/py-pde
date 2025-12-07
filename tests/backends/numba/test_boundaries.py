"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import UnitGrid
from pde.backends.numba._boundaries import make_virtual_point_evaluator
from pde.grids.boundaries.local import BCBase


def test_virtual_points_nb():
    """Test the calculation of virtual points."""
    g = UnitGrid([2])
    data = np.array([1, 2])

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
        ev = make_virtual_point_evaluator(bc)
        assert ev(data, (2,) if up else (-1,)) == pytest.approx(val)


@pytest.mark.parametrize("upper", [False, True])
def test_virtual_points_linked_data_nb(upper):
    """Test the calculation of virtual points with linked_data."""
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
    ev = make_virtual_point_evaluator(bc)
    assert ev(data, point) == pytest.approx(6)

    # test derivative boundary conditions (wrt to outwards derivative)
    bc = BCBase.from_data(g, 0, upper, {"type": "derivative", "value": bc_data})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 4

    assert bc.get_virtual_point(data, point) == pytest.approx(4)
    ev = make_virtual_point_evaluator(bc)
    assert ev(data, point) == pytest.approx(4)

    # test derivative boundary conditions (wrt to outwards derivative)
    bc = BCBase.from_data(g, 0, upper, {"type": "mixed", "value": bc_data, "const": 3})
    assert not bc.value_is_linked
    bc.link_value(bc_data)
    assert bc.value_is_linked
    bc_data[:] = 4

    assert bc.get_virtual_point(data, point) == pytest.approx(1)
    ev = make_virtual_point_evaluator(bc)
    assert ev(data, point) == pytest.approx(1)


def test_inhomogeneous_bcs_1d_nb():
    """Test inhomogeneous boundary conditions in 1d grids."""
    g = UnitGrid([2])
    data = np.ones((2,))  # field is 1 everywhere

    # first order bc
    bc_x = BCBase.from_data(g, 0, True, {"value": "x**2"})
    ev = make_virtual_point_evaluator(bc_x)
    assert ev(data, (1,)) == pytest.approx(7.0)

    # test lower bc
    bc_x = BCBase.from_data(g, 0, False, {"value": "x**2"})
    ev = make_virtual_point_evaluator(bc_x)
    assert ev(data, (0,)) == pytest.approx(-1.0)


def test_inhomogeneous_bcs_2d_nb():
    """Test inhomogeneous boundary conditions in 2d grids."""
    g = UnitGrid([2, 2])
    data = np.ones((2, 2))

    # second order bc
    bc_x = BCBase.from_data(g, 0, True, {"curvature": "y"})
    ev = make_virtual_point_evaluator(bc_x)
    assert ev(data, (1, 0)) == pytest.approx(1.5)
    assert ev(data, (1, 1)) == pytest.approx(2.5)
