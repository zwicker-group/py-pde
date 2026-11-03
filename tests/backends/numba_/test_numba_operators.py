"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np

from pde import ScalarField, UnitGrid


def test_numba_no_bc_operator():
    """Test applying an operator without bcs."""
    grid = UnitGrid(4)
    op = grid.make_operator_no_bc("laplace", backend="numba")
    field = ScalarField.random_uniform(grid)

    @nb.njit
    def f(arr):
        return op(arr)

    assert not any(np.isnan(f(field._data_full)))
