"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

import pde
from pde import MovieStorage
from pde.tools.misc import module_available


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg` module")
@pytest.mark.parametrize("dim", [1, 2])
def test_movie_storage_scalar_field(dim, tmp_path):
    """test writing scalar field"""
    path = tmp_path / f"test_movie_storage_scalar_{dim}.mp4"

    grid = pde.UnitGrid([8] * dim)
    field = pde.ScalarField(grid)
    eq = pde.PDE({"a": "1"})
    writer = MovieStorage(path, vmax=5)
    eq.solve(field, t_range=3.5, dt=0.1, backend="numpy", tracker=writer.tracker(1))

    reader = MovieStorage(path)
    assert len(reader) == 4
    for i, field in enumerate(reader):
        assert field.grid == grid
        np.testing.assert_allclose(field.data, i, rtol=0.2)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg` module")
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("num_fields", [1, 2, 3])
def test_movie_storage_scalar_fields(dim, num_fields, tmp_path):
    """test writing three scalar field"""
    path = tmp_path / f"test_movie_storage_scalar_{dim}_{num_fields}.mp4"

    grid = pde.UnitGrid([8] * dim)
    field = pde.FieldCollection([pde.ScalarField(grid)] * num_fields, copy_fields=True)
    eq = pde.PDE(dict([("a", "1"), ("b", "2"), ("c", "3")][:num_fields]))
    writer = MovieStorage(path, vmax=[5, 8, 12])
    eq.solve(field, t_range=3.5, dt=0.1, backend="numpy", tracker=writer.tracker(1))

    reader = MovieStorage(path)
    assert len(reader) == 4
    for i, fields in enumerate(reader):
        assert fields.grid == grid
        for j, field in enumerate(fields, 1):
            np.testing.assert_allclose(field.data, j * i)
