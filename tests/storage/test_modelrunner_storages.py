"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

import pde
from pde.tools.misc import module_available


@pytest.mark.skipif(
    not module_available("modelrunner"), reason="requires `py-modelrunner` package"
)
def test_storage_write_trajectory(tmp_path):
    """Test simple storage writing."""
    import modelrunner as mr

    path = tmp_path / "storage.json"
    storage = mr.open_storage(path, mode="truncate")

    field = pde.ScalarField.random_uniform(pde.UnitGrid([8, 8]))
    eq = pde.DiffusionPDE()
    eq.solve(
        field,
        t_range=2.5,
        dt=0.1,
        backend="numpy",
        tracker=pde.ModelrunnerStorage(storage).tracker(1),
    )

    assert path.is_file()
    assert len(pde.ModelrunnerStorage(path)) == 3
    np.testing.assert_allclose(pde.ModelrunnerStorage(path).times, [0, 1, 2])
