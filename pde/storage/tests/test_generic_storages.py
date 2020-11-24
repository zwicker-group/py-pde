"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import functools

import numpy as np
import pytest
from pde import DiffusionPDE, FileStorage, MemoryStorage, UnitGrid
from pde.fields import FieldCollection, ScalarField, Tensor2Field, VectorField
from pde.tools.misc import module_available


def test_storage_write(tmp_path):
    """ test simple memory storage """
    dim = 5
    grid = UnitGrid([dim])
    field = ScalarField(grid)

    storage_classes = {"MemoryStorage": MemoryStorage}
    if module_available("h5py"):
        file_path = tmp_path / "test_storage_write.hdf5"
        storage_classes["FileStorage"] = functools.partial(FileStorage, file_path)

    for name, storage_cls in storage_classes.items():
        storage = storage_cls(info={"a": 1})
        storage.start_writing(field, info={"b": 2})
        storage.append(field.copy(data=np.arange(dim)), 0)
        storage.append(field.copy(data=np.arange(dim)), 1)
        storage.end_writing()

        assert not storage.has_collection

        np.testing.assert_allclose(storage.times, np.arange(2))
        for f in storage:
            np.testing.assert_array_equal(f.data, np.arange(dim))
        for i in range(2):
            np.testing.assert_array_equal(storage[i].data, np.arange(dim))
        assert {"a": 1, "b": 2}.items() <= storage.info.items()

        storage = storage_cls()
        storage.clear()
        for i in range(3):
            storage.start_writing(field)
            storage.append(field.copy(data=np.arange(dim) + i), i)
            storage.end_writing()

        np.testing.assert_allclose(
            storage.times, np.arange(3), err_msg="storage class: " + name
        )


def test_storage_truncation(tmp_path):
    """ test whether simple trackers can be used """
    file = tmp_path / "test_storage_truncation.hdf5"
    for truncate in [True, False]:
        storages = [MemoryStorage()]
        if module_available("h5py"):
            storages.append(FileStorage(file))
        tracker_list = [s.tracker(interval=0.01) for s in storages]

        grid = UnitGrid([8, 8])
        state = ScalarField.random_uniform(grid, 0.2, 0.3)
        pde = DiffusionPDE()

        pde.solve(state, t_range=0.1, dt=0.001, tracker=tracker_list)
        if truncate:
            for storage in storages:
                storage.clear()
        pde.solve(state, t_range=[0.1, 0.2], dt=0.001, tracker=tracker_list)

        times = np.arange(0.1, 0.201, 0.01)
        if not truncate:
            times = np.r_[np.arange(0, 0.101, 0.01), times]
        for storage in storages:
            msg = f"truncate={truncate}, storage={storage}"
            np.testing.assert_allclose(storage.times, times, err_msg=msg)

        assert not storage.has_collection


def test_storing_extract_range(tmp_path):
    """ test methods specific to FieldCollections in memory storage """
    sf = ScalarField(UnitGrid([1]))

    storage_classes = {"MemoryStorage": MemoryStorage}
    if module_available("h5py"):
        file_path = tmp_path / "test_storage_write.hdf5"
        storage_classes["FileStorage"] = functools.partial(FileStorage, file_path)

    for storage_cls in storage_classes.values():
        # store some data
        s1 = storage_cls()
        s1.start_writing(sf)
        s1.append(sf.copy(data=np.array([0])), 0)
        s1.append(sf.copy(data=np.array([2])), 1)
        s1.end_writing()

        np.testing.assert_equal(s1[0].data, 0)
        np.testing.assert_equal(s1[1].data, 2)
        np.testing.assert_equal(s1[-1].data, 2)
        np.testing.assert_equal(s1[-2].data, 0)

        with pytest.raises(IndexError):
            s1[2]
        with pytest.raises(IndexError):
            s1[-3]

        # test extraction
        s2 = s1.extract_time_range()
        assert s2.times == list(s1.times)
        np.testing.assert_allclose(s2.data, s1.data)
        s3 = s1.extract_time_range(0.5)
        assert s3.times == s1.times[:1]
        np.testing.assert_allclose(s3.data, s1.data[:1])
        s4 = s1.extract_time_range((0.5, 1.5))
        assert s4.times == s1.times[1:]
        np.testing.assert_allclose(s4.data, s1.data[1:])


def test_storing_collection(tmp_path):
    """ test methods specific to FieldCollections in memory storage """
    grid = UnitGrid([2, 2])
    f1 = ScalarField.random_uniform(grid, 0.1, 0.4)
    f2 = VectorField.random_uniform(grid, 0.1, 0.4)
    f3 = Tensor2Field.random_uniform(grid, 0.1, 0.4)
    fc = FieldCollection([f1, f2, f3])

    storage_classes = {"MemoryStorage": MemoryStorage}
    if module_available("h5py"):
        file_path = tmp_path / "test_storage_write.hdf5"
        storage_classes["FileStorage"] = functools.partial(FileStorage, file_path)

    for storage_cls in storage_classes.values():
        # store some data
        storage = storage_cls()
        storage.start_writing(fc)
        storage.append(fc, 0)
        storage.append(fc, 1)
        storage.end_writing()

        assert storage.has_collection
        assert storage.extract_field(0)[0] == f1
        assert storage.extract_field(1)[0] == f2
        assert storage.extract_field(2)[0] == f3


def test_storage_apply(tmp_path):
    """ test the apply function of StorageBase """
    grid = UnitGrid([2])
    field = ScalarField(grid)

    storage_classes = {"None": None, "MemoryStorage": MemoryStorage}
    if module_available("h5py"):
        file_path = tmp_path / "test_storage_apply.hdf5"
        storage_classes["FileStorage"] = functools.partial(FileStorage, file_path)

    s1 = MemoryStorage()
    s1.start_writing(field, info={"b": 2})
    s1.append(field.copy(data=np.array([0, 1])), 0)
    s1.append(field.copy(data=np.array([1, 2])), 1)
    s1.end_writing()

    for name, storage_cls in storage_classes.items():
        out = None if storage_cls is None else storage_cls()
        s2 = s1.apply(lambda x: x + 1, out=out)
        assert storage_cls is None or s2 is out
        assert len(s2) == 2
        np.testing.assert_allclose(s2.times, s1.times)
        assert s2[0] == ScalarField(grid, [1, 2]), name
        assert s2[1] == ScalarField(grid, [2, 3]), name

    # test empty storage
    s1 = MemoryStorage()
    s2 = s1.apply(lambda x: x + 1)
    assert len(s2) == 0


def test_storage_copy(tmp_path):
    """ test the copy function of StorageBase """
    grid = UnitGrid([2])
    field = ScalarField(grid)

    storage_classes = {"None": None, "MemoryStorage": MemoryStorage}
    if module_available("h5py"):
        file_path = tmp_path / "test_storage_apply.hdf5"
        storage_classes["FileStorage"] = functools.partial(FileStorage, file_path)

    s1 = MemoryStorage()
    s1.start_writing(field, info={"b": 2})
    s1.append(field.copy(data=np.array([0, 1])), 0)
    s1.append(field.copy(data=np.array([1, 2])), 1)
    s1.end_writing()

    for name, storage_cls in storage_classes.items():
        out = None if storage_cls is None else storage_cls()
        s2 = s1.copy(out=out)
        assert storage_cls is None or s2 is out
        assert len(s2) == 2
        np.testing.assert_allclose(s2.times, s1.times)
        assert s2[0] == s1[0], name
        assert s2[1] == s1[1], name

    # test empty storage
    s1 = MemoryStorage()
    s2 = s1.copy()
    assert len(s2) == 0
