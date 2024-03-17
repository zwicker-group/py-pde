"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

import pde
from pde import MovieStorage
from pde.tools.misc import module_available


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("dim", [1, 2])
def test_movie_storage_scalar(dim, tmp_path, rng):
    """test storing scalar field as movie"""
    path = tmp_path / f"test_movie_storage_scalar.mov"

    grid = pde.UnitGrid([16] * dim)
    field = pde.ScalarField.random_uniform(grid, rng=rng)
    eq = pde.DiffusionPDE()
    writer = MovieStorage(path)
    storage = pde.MemoryStorage()
    eq.solve(
        field,
        t_range=3.5,
        dt=0.1,
        backend="numpy",
        tracker=[storage.tracker(2), writer.tracker(2)],
    )

    reader = MovieStorage(path)
    assert len(reader) == 2
    np.testing.assert_allclose(reader.times, [0, 2])
    for i, field in enumerate(reader):
        assert field.grid == grid
        np.testing.assert_allclose(field.data, storage[i].data, atol=0.1)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("num_fields", [1, 2, 3])
def test_movie_storage_collection(dim, num_fields, tmp_path):
    """test storing field collection as movie"""
    path = tmp_path / f"test_movie_storage_collection_{dim}_{num_fields}.mp4"

    grid = pde.UnitGrid([8] * dim)
    field = pde.FieldCollection([pde.ScalarField(grid)] * num_fields, copy_fields=True)
    eq = pde.PDE(dict([("a", "1"), ("b", "2"), ("c", "3")][:num_fields]))
    writer = MovieStorage(path, vmax=[5, 10, 15])
    storage = pde.MemoryStorage()
    eq.solve(
        field,
        t_range=3.5,
        dt=0.1,
        backend="numpy",
        tracker=[storage.tracker(1), writer.tracker(1)],
    )

    reader = MovieStorage(path)
    assert len(reader) == 4
    assert reader.vmax == [5, 10, 15]
    for t, fields in enumerate(reader):
        assert fields.grid == grid
        for i, field in enumerate(fields):
            np.testing.assert_allclose(field.data, (i + 1) * t, atol=0.1, rtol=0.1)
            # np.testing.assert_allclose(field.data, storage[t][i].data)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("dim", [1, 2])
def test_movie_storage_vector(dim, tmp_path, rng):
    """test storing scalar field as movie"""
    path = tmp_path / f"test_movie_storage_vector.mov"

    grid = pde.UnitGrid([16] * dim)
    field = pde.VectorField.random_uniform(grid, rng=rng)
    eq = pde.PDE({"a": "-0.5 * a"})
    writer = MovieStorage(path)
    storage = pde.MemoryStorage()
    eq.solve(
        field,
        t_range=3.5,
        dt=0.1,
        backend="numpy",
        tracker=[storage.tracker(2), writer.tracker(2)],
    )
    assert len(writer._norms) == dim

    reader = MovieStorage(path)
    assert len(reader) == 2
    np.testing.assert_allclose(reader.times, [0, 2])
    for i, field in enumerate(reader):
        assert field.grid == grid
        assert isinstance(field, pde.VectorField)
        assert field.data.shape == (dim,) + grid.shape
        np.testing.assert_allclose(field.data, storage[i].data, atol=0.1)


# test all pixel formats
# test failure of complex data
