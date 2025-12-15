"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

import pde


def test_adams_bashforth():
    """Test the adams_bashforth method."""
    eq = pde.PDE({"y": "y"})
    state = pde.ScalarField(pde.UnitGrid([1]), 1)
    storage = pde.MemoryStorage()
    eq.solve(
        state,
        t_range=2.1,
        dt=0.5,
        solver="adams–bashforth",
        tracker=storage.tracker(0.5),
    )
    np.testing.assert_allclose(
        np.ravel([f.data for f in storage]),
        [1, 13 / 8, 83 / 32, 529 / 128, 3371 / 512, 21481 / 2048],
    )
