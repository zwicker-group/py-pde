"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde import CylindricalSymGrid, PolarSymGrid, SphericalSymGrid, UnitGrid
from pde.backends.numba.grids import get_grid_numba_type


def test_grid_numba_type():
    """Test the `get_grid_numba_type` function."""
    assert get_grid_numba_type(UnitGrid(4)) == "f8[:]"
    assert get_grid_numba_type(UnitGrid([4, 4])) == "f8[:, :]"
    assert get_grid_numba_type(UnitGrid([4, 4, 4])) == "f8[:, :, :]"
    assert get_grid_numba_type(PolarSymGrid(4, 8)) == "f8[:]"
    assert get_grid_numba_type(SphericalSymGrid(4, 8)) == "f8[:]"
    assert get_grid_numba_type(CylindricalSymGrid(4, (-1, 2), (8, 9))) == "f8[:, :]"
