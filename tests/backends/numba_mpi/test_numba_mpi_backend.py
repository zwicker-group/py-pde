"""Test general backend selection.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from pde import CartesianGrid, ScalarField, get_backend


def test_numba_mpi_backend_fallback():
    """Test basic methods of the numba_mpi backend."""
    backend = get_backend("numba_mpi")

    assert "laplace" in backend.get_registered_operators(CartesianGrid)

    # check operators with BCs
    grid = CartesianGrid([[0, 1]], 4, periodic=True)
    field = ScalarField(grid, 1)
    op = grid.make_operator("laplace", bc="periodic", backend=backend)
    res = op(field.data)
    np.testing.assert_allclose(res, 0)

    # check operators without BCs
    op_no_bc = grid.make_operator_no_bc("laplace", backend=backend)
    field._data_full[:] = 1
    res = field.copy()
    op_no_bc(field._data_full, out=res.data)
    np.testing.assert_allclose(res.data, 0)
