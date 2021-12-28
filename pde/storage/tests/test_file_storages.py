"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

import pde
from pde import DiffusionPDE, FileStorage, ScalarField, UnitGrid
from pde.tools.misc import skipUnlessModule


@skipUnlessModule("h5py")
@pytest.mark.parametrize("collection", [True, False])
def test_storage_persistence(collection, tmp_path):
    """test writing to persistent trackers"""
    dim = 5
    grid = UnitGrid([dim])
    scalar = ScalarField(grid)
    vector = pde.VectorField(grid)
    if collection:
        state = pde.FieldCollection([scalar, vector])
    else:
        state = scalar

    def assert_storage_content(storage, expect):
        """helper function testing storage content"""
        if collection:
            for i in range(2):
                field_data = storage.extract_field(i).data
                np.testing.assert_array_equal(np.ravel(field_data), expect)
        else:
            np.testing.assert_array_equal(np.ravel(storage.data), expect)

    path = tmp_path / f"test_storage_persistence_{collection}.hdf5"

    # write some data
    for write_mode in ["append", "truncate_once", "truncate"]:
        with FileStorage(path, info={"a": 1}, write_mode=write_mode) as writer:

            # first batch
            writer.start_writing(state, info={"b": 2})
            scalar.data = np.arange(dim)
            vector.data[:] = np.arange(dim)
            writer.append(state, 0)
            scalar.data = np.arange(dim, 2 * dim)
            vector.data[:] = np.arange(dim, 2 * dim)
            writer.append(state)
            writer.end_writing()

            # read first batch
            np.testing.assert_array_equal(writer.times, np.arange(2))
            assert_storage_content(writer, np.arange(10))
            assert {"a": 1, "b": 2}.items() <= writer.info.items()

            # second batch
            writer.start_writing(state, info={"c": 3})
            scalar.data = np.arange(2 * dim, 3 * dim)
            vector.data[:] = np.arange(2 * dim, 3 * dim)
            writer.append(state, 2)
            writer.end_writing()

        # read the data
        with FileStorage(path) as reader:
            if write_mode == "truncate":
                np.testing.assert_array_equal(reader.times, np.array([2]))
                assert_storage_content(reader, np.arange(10, 15))
                assert reader.shape == (1, 2, 5) if collection else (1, 5)
                info = {"c": 3}
                assert info.items() <= reader.info.items()

            else:
                np.testing.assert_array_equal(reader.times, np.arange(3))
                assert_storage_content(reader, np.arange(15))
                assert reader.shape == (3, 2, 5) if collection else (3, 5)
                info = {"a": 1, "b": 2, "c": 3}
                assert info.items() <= reader.info.items()


@skipUnlessModule("h5py")
@pytest.mark.parametrize("compression", [True, False])
def test_simulation_persistence(compression, tmp_path):
    """test whether a tracker can accurately store information about simulation"""
    path = tmp_path / "test_simulation_persistence.hdf5"
    storage = FileStorage(path, compression=compression)

    # write some simulation data
    pde = DiffusionPDE()
    grid = UnitGrid([16, 16])  # generate grid
    state = ScalarField.random_uniform(grid, 0.2, 0.3)
    pde.solve(state, t_range=0.11, dt=0.001, tracker=storage.tracker(interval=0.05))
    storage.close()

    # read the data
    storage = FileStorage(path)
    np.testing.assert_almost_equal(storage.times, [0, 0.05, 0.1])
    data = np.array(storage.data)
    assert data.shape == (3,) + state.data.shape
    grid_res = storage.grid
    assert grid == grid_res
    grid_res = storage.grid
    assert grid == grid_res


@skipUnlessModule("h5py")
@pytest.mark.parametrize("compression", [True, False])
def test_storage_fixed_size(compression, tmp_path):
    """test setting fixed size of FileStorage objects"""
    c = ScalarField(UnitGrid([2]), data=1)

    for fixed in [True, False]:
        path = tmp_path / f"test_storage_fixed_size_{fixed}.hdf5"
        storage = FileStorage(
            path, max_length=1 if fixed else None, compression=compression
        )
        assert len(storage) == 0

        storage.start_writing(c)
        assert len(storage) == 0
        storage.append(c, 0)
        assert len(storage) == 1

        if fixed:
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                storage.append(c, 1)
            assert len(storage) == 1
            np.testing.assert_allclose(storage.times, [0])
        else:
            storage.append(c, 1)
            assert len(storage) == 2
            np.testing.assert_allclose(storage.times, [0, 1])


@skipUnlessModule("h5py")
def test_appending(tmp_path):
    """test the appending data"""
    path = tmp_path / "test_appending.hdf5"

    c = ScalarField(UnitGrid([2]), data=1)
    storage = FileStorage(path)
    storage.start_writing(c)
    assert len(storage) == 0
    storage.append(c, 0)
    assert storage._file_state == "writing"
    assert len(storage) == 1
    storage.close()

    storage2 = FileStorage(path, write_mode="append")
    storage2.start_writing(c)
    storage2.append(c, 1)
    storage2.close()

    assert len(storage2) == 2


@skipUnlessModule("h5py")
def test_keep_opened(tmp_path):
    """test the keep opened option"""
    path = tmp_path / "test_keep_opened.hdf5"

    c = ScalarField(UnitGrid([2]), data=1)
    storage = FileStorage(path, keep_opened=False)
    storage.start_writing(c)
    assert len(storage) == 0
    storage.append(c, 0)
    assert storage._file_state == "closed"
    assert len(storage) == 1
    assert storage._file_state == "reading"
    storage.append(c, 1)
    assert len(storage) == 2

    storage2 = FileStorage(path, write_mode="append")
    assert storage.times == storage2.times
    assert storage.data == storage2.data
    storage.close()  # close the old storage to enable writing here
    storage2.start_writing(c)
    storage2.append(c, 2)
    storage2.close()

    assert len(storage2) == 3
    np.testing.assert_allclose(storage2.times, np.arange(3))


@skipUnlessModule("h5py")
@pytest.mark.parametrize("dtype", [bool, float, complex])
def test_write_types(dtype, tmp_path):
    """test whether complex data can be written"""
    path = tmp_path / "test_type_writing.hdf5"

    grid = UnitGrid([32])
    c = ScalarField.random_uniform(grid).copy(dtype=dtype)
    if dtype == complex:
        c += 1j * ScalarField.random_uniform(grid)

    storage = FileStorage(path, keep_opened=False)
    storage.start_writing(c)
    assert len(storage) == 0
    storage.append(c, 0)
    assert storage._file_state == "closed"
    assert len(storage) == 1
    assert storage._file_state == "reading"
    storage.append(c, 1)
    assert len(storage) == 2
    assert storage.dtype == np.dtype(dtype)

    storage2 = FileStorage(path, write_mode="append")
    assert storage.times == storage2.times
    assert storage.data == storage2.data
    storage.close()  # close the old storage to enable writing here
    storage2.start_writing(c)
    storage2.append(c, 2)
    storage2.close()

    assert len(storage2) == 3
    np.testing.assert_allclose(storage2.times, np.arange(3))
    assert storage2.dtype == np.dtype(dtype)

    storage3 = FileStorage(path, write_mode="reading")
    assert len(storage3) == 3
    for field in storage3:
        np.testing.assert_allclose(field.data, c.data)
    assert storage3.dtype == np.dtype(dtype)
