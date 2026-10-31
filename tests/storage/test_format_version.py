"""Test the handling of format versions of stored files."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pde
from pde.tools.misc import module_available

needs_h5py = pytest.mark.skipif(not module_available("h5py"), reason="requires `h5py`")
needs_av = pytest.mark.skipif(not module_available("av"), reason="requires `av`")


def _write(storage) -> None:
    grid = pde.UnitGrid([4])
    storage.start_writing(pde.ScalarField(grid, 0.5), info={"payload": "x"})
    for t in [0, 1]:
        storage.append(pde.ScalarField(grid, 0.5), t)
    storage.end_writing()


@needs_h5py
def test_file_storage_version(tmp_path):
    """Test that the format version is written and not part of the info."""
    import h5py

    path = tmp_path / "test.hdf5"
    _write(pde.FileStorage(path))

    with h5py.File(path, "r") as f:
        assert json.loads(f.attrs["version"]) == 1

    reader = pde.FileStorage(path)
    assert len(reader) == 2
    assert "version" not in reader.info
    assert reader.info["payload"] == "x"


@needs_h5py
def test_file_storage_missing_version(tmp_path):
    """Test that files without version information are read as version 1."""
    import h5py

    path = tmp_path / "test.hdf5"
    _write(pde.FileStorage(path))
    with h5py.File(path, "r+") as f:
        del f.attrs["version"]

    reader = pde.FileStorage(path)
    assert len(reader) == 2
    np.testing.assert_allclose(reader[1].data, 0.5)
    reader.close()

    # appending to such a file stores the version
    writer = pde.FileStorage(path, write_mode="append")
    writer.start_writing(pde.ScalarField(pde.UnitGrid([4]), 0.5))
    writer.append(pde.ScalarField(pde.UnitGrid([4]), 0.5), 2)
    writer.end_writing()
    writer.close()
    with h5py.File(path, "r") as f:
        assert json.loads(f.attrs["version"]) == 1
    assert len(pde.FileStorage(path)) == 3


@needs_h5py
def test_file_storage_unknown_version(tmp_path, monkeypatch):
    """Test that files with unknown format versions are rejected."""
    path = tmp_path / "test.hdf5"
    with monkeypatch.context() as m:
        m.setattr(pde.FileStorage, "_format_version", 2)
        _write(pde.FileStorage(path))

    with pytest.raises(RuntimeError, match="format version"):
        len(pde.FileStorage(path))
    with pytest.raises(RuntimeError, match="format version"):
        pde.FileStorage(path, write_mode="append")


@needs_av
def test_movie_storage_version(tmp_path):
    """Test that the format version is written and not part of the info."""
    import av

    path = tmp_path / "test.avi"
    _write(pde.MovieStorage(path))

    with av.open(str(path)) as container:
        comment = json.loads(
            container.metadata.get("comment", container.metadata.get("COMMENT", "{}"))
        )
    assert comment["version"] == 1

    reader = pde.MovieStorage(path)
    assert len(reader) == 2
    assert "version" not in reader.info
    assert reader.info["payload"] == "x"


@needs_av
def test_movie_storage_missing_version(tmp_path, monkeypatch):
    """Test that movies without version information are read as version 1."""
    path = tmp_path / "test.avi"
    original = pde.MovieStorage._get_metadata

    def get_metadata_without_version(self):
        info = json.loads(original(self))
        del info["version"]
        return json.dumps(info)

    monkeypatch.setattr(pde.MovieStorage, "_get_metadata", get_metadata_without_version)
    _write(pde.MovieStorage(path))
    monkeypatch.undo()

    reader = pde.MovieStorage(path)
    assert len(reader) == 2
    np.testing.assert_allclose(reader[0].data, 0.5, atol=1e-4)


@needs_av
def test_movie_storage_unknown_version(tmp_path, monkeypatch):
    """Test that movies with unknown format versions are rejected."""
    path = tmp_path / "test.avi"
    with monkeypatch.context() as m:
        m.setattr(pde.MovieStorage, "_format_version", 2)
        _write(pde.MovieStorage(path))

    with pytest.raises(RuntimeError, match="format version"):
        pde.MovieStorage(path).times  # noqa: B018
