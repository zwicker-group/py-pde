"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
from pathlib import Path

import numpy as np
import pytest

import pde
from pde import FileStorage, MovieStorage
from pde.tools.ffmpeg import formats
from pde.tools.misc import module_available

RESOURCES_PATH = Path(__file__).resolve().parent / "resources"


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("dim", [1, 2])
def test_movie_storage_scalar(dim, tmp_path, rng):
    """Test storing scalar field as movie."""
    path = tmp_path / "test_movie_storage_scalar.avi"

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
        np.testing.assert_allclose(field.data, storage[i].data, atol=0.01)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("num_fields", [1, 2, 3])
def test_movie_storage_collection(dim, num_fields, tmp_path):
    """Test storing field collection as movie."""
    path = tmp_path / f"test_movie_storage_collection_{dim}_{num_fields}.avi"

    grid = pde.UnitGrid([8] * dim)
    field = pde.FieldCollection([pde.ScalarField(grid)] * num_fields, copy_fields=True)
    eq = pde.PDE(dict([("a", "1"), ("b", "2"), ("c", "3")][:num_fields]))
    writer = MovieStorage(path, vmax=[5, 10, 15])
    eq.solve(field, t_range=3.5, dt=0.1, backend="numpy", tracker=writer.tracker(1))

    reader = MovieStorage(path)
    assert len(reader) == 4
    assert reader.vmax == [5, 10, 15]
    for t, fields in enumerate(reader):
        assert fields.grid == grid
        for i, field in enumerate(fields):
            np.testing.assert_allclose(field.data, (i + 1) * t, atol=0.02, rtol=0.02)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("dim", [1, 2])
def test_movie_storage_vector(dim, tmp_path, rng):
    """Test storing scalar field as movie."""
    path = tmp_path / "test_movie_storage_vector.avi"

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
        np.testing.assert_allclose(field.data, storage[i].data, atol=0.01)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("ext", [".mov", ".avi", ".mkv"])
def test_movie_storage_containers(ext, tmp_path, rng):
    """Test storing scalar field as movie with different extensions."""
    path = tmp_path / f"test_movie_storage_scalar{ext}"

    grid = pde.UnitGrid([16])
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
        np.testing.assert_allclose(field.data, storage[i].data, atol=0.01)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("name,video_format", formats.items())
def test_video_format(name, video_format, tmp_path, rng):
    """Test all video_formats."""
    if np.issubdtype(video_format.dtype, np.integer):
        assert video_format.max_value == np.iinfo(video_format.dtype).max
    assert np.dtype(video_format.dtype).itemsize == video_format.bytes_per_channel

    path = tmp_path / f"test_movie_storage_{name}.avi"

    field = pde.ScalarField.random_uniform(pde.UnitGrid([16]), rng=rng)
    writer = MovieStorage(path, video_format=name)
    storage = pde.MemoryStorage()
    pde.DiffusionPDE().solve(
        field,
        t_range=3.5,
        dt=0.1,
        backend="numpy",
        tracker=[storage.tracker(2), writer.tracker(2)],
    )

    atol = {8: 1e-2, 16: 1e-4, 32: 1e-8, 64: 1e-15}[video_format.bits_per_channel]
    reader = MovieStorage(path)
    for i, field in enumerate(reader):
        np.testing.assert_allclose(field.data, storage[i].data, atol=atol)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
def test_too_many_channels(tmp_path, rng):
    """Test that data with too many channels throws an error."""
    path = tmp_path / "test_movie_complex.avi"

    field = pde.FieldCollection.scalar_random_uniform(5, pde.UnitGrid([16]), rng=rng)
    writer = MovieStorage(path)
    eq = pde.PDE(dict.fromkeys("abcde", "0"))
    with pytest.raises(RuntimeError):
        eq.solve(field, t_range=3.5, backend="numpy", tracker=writer.tracker(2))


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
def test_complex_data(tmp_path, rng):
    """Test that complex data throws an error."""
    path = tmp_path / "test_movie_complex.avi"

    field = pde.ScalarField.random_uniform(pde.UnitGrid([16]), dtype=complex, rng=rng)
    writer = MovieStorage(path)
    with pytest.raises(NotImplementedError):
        pde.DiffusionPDE().solve(
            field, t_range=3.5, backend="numpy", tracker=writer.tracker(2)
        )


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
def test_wrong_format():
    """Test how wrong files are dealt with."""
    from ffmpeg._run import Error as FFmpegError

    reader = MovieStorage(RESOURCES_PATH / "does_not_exist.avi")
    with pytest.raises(OSError):
        print(reader.times)

    reader = MovieStorage(RESOURCES_PATH / "empty.avi")
    with pytest.raises(FFmpegError):
        print(reader.times)

    reader = MovieStorage(RESOURCES_PATH / "no_metadata.avi")
    np.testing.assert_allclose(reader.times, [0, 1])
    with pytest.raises(RuntimeError):
        reader[0]


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize("path", RESOURCES_PATH.glob("*.hdf5"))
def test_stored_files(path):
    """Test stored files."""
    file_reader = FileStorage(path)
    movie_reader = MovieStorage(path.with_suffix(".avi"))

    np.testing.assert_allclose(file_reader.times, movie_reader.times)
    assert file_reader.info["payload"] == movie_reader.info["payload"]
    np.testing.assert_allclose(file_reader.times, movie_reader.times)
    for a, b in zip(file_reader, movie_reader):
        assert a.grid == b.grid
        np.testing.assert_allclose(a.data, b.data, atol=1e-4)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
@pytest.mark.parametrize(
    "interrupt, expected",
    [
        (pde.FixedInterrupts([0.5, 0.7, 1.4]), [0.5, 0.7, 1.4]),
        (pde.ConstantInterrupts(1, t_start=1), [1, 2]),
        (2, [0, 2]),
        (pde.LogarithmicInterrupts(0.5, 2), [0, 0.5, 1.5]),
    ],
)
def test_stored_times(interrupt, expected, tmp_path):
    """Test how times are stored."""
    path = tmp_path / "test_movie_times.avi"

    field = pde.ScalarField(pde.UnitGrid([3]))
    writer = MovieStorage(path, write_times=True)
    eq = pde.DiffusionPDE()
    eq.solve(
        field, t_range=2.1, dt=0.1, backend="numpy", tracker=writer.tracker(interrupt)
    )

    assert writer._filename_times.exists()

    reader = MovieStorage(path, write_mode="reading")
    np.testing.assert_allclose(reader.times, expected, atol=0.01)


@pytest.mark.skipif(not module_available("ffmpeg"), reason="requires `ffmpeg-python`")
def test_unequal_spaced_times(tmp_path, caplog):
    """Test whether a warning is generated for unequally spaced times."""
    path = tmp_path / "test_movie_unequal_times.avi"

    field = pde.ScalarField(pde.UnitGrid([3]))
    writer = MovieStorage(path, write_times=False)
    tracker = writer.tracker(pde.FixedInterrupts([0.5, 0.7, 1.4]))
    eq = pde.DiffusionPDE()
    with caplog.at_level(logging.WARNING):
        eq.solve(field, t_range=2.1, dt=0.1, backend="numpy", tracker=tracker)

    assert "write_times=True" in caplog.text
    assert "Time mismatch" in caplog.text

    reader = MovieStorage(path, write_mode="reading")
    np.testing.assert_allclose(reader.times, [0, 1, 2], atol=0.01)
