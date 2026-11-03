"""Test whether stored files of previous versions can still be read.

Files of older versions are located in subfolders of `tests/resources/FileStorage` and
`tests/resources/MovieStorage`, see the README files there. The expected content is
defined analytically in `_compat_cases.py`. In addition, the files written by the
current version are checked in the same way, which tests the generator scripts.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest
from _compat_cases import CASES, RESOURCES_ROOT, Case

import pde
from pde.tools.ffmpeg import formats
from pde.tools.misc import module_available

if TYPE_CHECKING:
    from pathlib import Path

SKIP = {
    "FileStorage": pytest.mark.skipif(
        not module_available("h5py"), reason="requires `h5py`"
    ),
    "MovieStorage": pytest.mark.skipif(
        not module_available("av"), reason="requires `av`"
    ),
}


def _version_folders(storage_class: str) -> list[Path]:
    """Find folders with stored files, ignoring `current` and non-version folders."""
    root = RESOURCES_ROOT / storage_class
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("v[0-9]*") if p.is_dir())


def _params():
    """Create test parameters for all storage classes, version folders, and cases."""
    for storage_class, cases in CASES.items():
        folders: list[Path | None] = [None, *_version_folders(storage_class)]
        for folder in folders:
            for case in cases:
                if folder is not None and not (folder / case.filename).exists():
                    continue  # version might not have this case
                name = "generated" if folder is None else folder.name
                yield pytest.param(
                    storage_class,
                    folder,
                    case,
                    id=f"{storage_class}-{name}-{case.name}",
                    marks=SKIP[storage_class],
                )


def _open_reader(case: Case, path: Path) -> pde.StorageBase:
    if case.storage_class == "FileStorage":
        return pde.FileStorage(path)
    return pde.MovieStorage(path)


def _check_case(case: Case, path: Path) -> None:
    """Check that the stored file `path` has the content expected for `case`."""
    reader = _open_reader(case, path)

    # time information
    assert len(reader) == len(case.times)
    np.testing.assert_allclose(reader.times, case.times)

    # meta data
    assert reader.info["payload"] == case.name
    if case.storage_class == "MovieStorage":
        video_format = formats[reader.info["video_format"]]
        assert video_format.bits_per_channel == case.storage_kwargs["bits_per_channel"]
        assert reader.info.get("write_times", False) == (
            case.storage_kwargs.get("write_times", False)
        )

    # grid
    assert reader.grid == case.make_grid()
    assert reader.grid.periodic == case.make_grid().periodic

    # data, accessed via different interfaces
    expected = [case.make_field(t) for t in case.times]
    if case.storage_class == "FileStorage":
        assert reader.data.dtype == np.dtype(case.dtype)

    def check(field, expect):
        assert type(field) is type(expect)
        assert field.grid == expect.grid
        assert field.label == expect.label
        if isinstance(field, pde.FieldCollection):
            assert [f.label for f in field] == [f.label for f in expect]
            assert [type(f) for f in field] == [type(f) for f in expect]
        if case.storage_class == "FileStorage":
            assert field.dtype == expect.dtype
        np.testing.assert_allclose(field.data, expect.data, atol=case.atol, rtol=0)

    for i, expect in enumerate(expected):
        check(reader[i], expect)  # random access
    for field, expect in zip(reader, expected, strict=True):
        check(field, expect)  # iteration
    for (time, field), t, expect in zip(
        reader.items(), case.times, expected, strict=True
    ):
        assert time == pytest.approx(t)
        check(field, expect)

    if case.storage_class == "FileStorage":
        # the raw data of movies is stored as frames, so only compare for files
        np.testing.assert_allclose(
            reader.data, np.array([e.data for e in expected]), atol=0, rtol=0
        )


@pytest.mark.parametrize(("storage_class", "folder", "case"), list(_params()))
def test_stored_files(storage_class, folder, case, tmp_path):
    """Test that stored files can be read and contain the expected data."""
    if folder is None:
        # check files written by the current version
        case.write(tmp_path)
    else:
        # work on copies so that the stored resources are never modified
        for file in folder.glob(f"{case.filename}*"):
            shutil.copy(file, tmp_path / file.name)
    _check_case(case, tmp_path / case.filename)


@pytest.mark.skipif(not module_available("h5py"), reason="requires `h5py`")
def test_file_storage_stays_unmodified(tmp_path):
    """Test that reading stored files does not modify them."""
    case = CASES["FileStorage"][0]
    path = case.write(tmp_path)
    before = path.read_bytes()
    _check_case(case, path)
    assert path.read_bytes() == before
