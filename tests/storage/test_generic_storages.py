"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import functools
import platform

import numpy as np
import pytest

from pde import DiffusionPDE, FileStorage, MemoryStorage, UnitGrid
from pde.fields import FieldCollection, ScalarField, Tensor2Field, VectorField
from pde.tools import mpi
from pde.tools.misc import module_available

STORAGE_CLASSES = [MemoryStorage, FileStorage]


@pytest.fixture
def storage_factory(tmp_path, storage_class):
    """helper fixture that provides a storage factory that initializes files"""
    if storage_class is FileStorage:
        # provide factory that initializes a FileStorage with a file
        if not module_available("h5py"):
            pytest.skip("No module `h5py`")
        file_path = tmp_path / "test_storage_write.hdf5"
        return functools.partial(FileStorage, file_path)

    # simply return the storage class assuming it is a factory function already
    return storage_class


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES)
def test_storage_write(storage_factory):
    """test simple memory storage"""
    dim = 5
    grid = UnitGrid([dim])
    field = ScalarField(grid)

    storage = storage_factory(info={"a": 1})
    storage.start_writing(field, info={"b": 2})
    field.data = np.arange(dim)
    storage.append(field, 0)
    field.data = np.arange(dim)
    storage.append(field, 1)
    storage.end_writing()

    assert not storage.has_collection

    np.testing.assert_allclose(storage.times, np.arange(2))
    for f in storage:
        np.testing.assert_array_equal(f.data, np.arange(dim))
    for i in range(2):
        np.testing.assert_array_equal(storage[i].data, np.arange(dim))
    assert {"a": 1, "b": 2}.items() <= storage.info.items()

    storage = storage_factory()
    storage.clear()
    for i in range(3):
        storage.start_writing(field)
        field.data = np.arange(dim) + i
        storage.append(field, i)
        storage.end_writing()

    np.testing.assert_allclose(storage.times, np.arange(3))


def test_storage_truncation(tmp_path, rng):
    """test whether simple trackers can be used"""
    file = tmp_path / "test_storage_truncation.hdf5"
    for truncate in [True, False]:
        storages = [MemoryStorage()]
        if module_available("h5py"):
            storages.append(FileStorage(file))
        tracker_list = [s.tracker(interrupts=0.01) for s in storages]

        grid = UnitGrid([8, 8])
        state = ScalarField.random_uniform(grid, 0.2, 0.3, rng=rng)
        eq = DiffusionPDE()

        eq.solve(state, t_range=0.1, dt=0.001, tracker=tracker_list)
        if truncate:
            for storage in storages:
                storage.clear()
        eq.solve(state, t_range=[0.1, 0.2], dt=0.001, tracker=tracker_list)

        times = np.arange(0.1, 0.201, 0.01)
        if not truncate:
            times = np.r_[np.arange(0, 0.101, 0.01), times]
        for storage in storages:
            msg = f"truncate={truncate}, storage={storage}"
            np.testing.assert_allclose(storage.times, times, err_msg=msg)

        if any(platform.win32_ver()):
            for storage in storages:
                if isinstance(storage, FileStorage):
                    storage.close()

        assert not storage.has_collection


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES)
def test_storing_extract_range(storage_factory):
    """test methods specific to FieldCollections in memory storage"""
    sf = ScalarField(UnitGrid([1]))

    # store some data
    s1 = storage_factory()
    s1.start_writing(sf)
    sf.data = np.array([0])
    s1.append(sf, 0)
    sf.data = np.array([2])
    s1.append(sf, 1)
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


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES)
def test_storing_collection(storage_factory, rng):
    """test methods specific to FieldCollections in memory storage"""
    grid = UnitGrid([2, 2])
    f1 = ScalarField.random_uniform(grid, 0.1, 0.4, label="a", rng=rng)
    f2 = VectorField.random_uniform(grid, 0.1, 0.4, label="b", rng=rng)
    f3 = Tensor2Field.random_uniform(grid, 0.1, 0.4, label="c", rng=rng)
    fc = FieldCollection([f1, f2, f3])

    # store some data
    storage = storage_factory()
    storage.start_writing(fc)
    storage.append(fc, 0)
    storage.append(fc, 1)
    storage.end_writing()

    assert storage.has_collection
    assert storage.extract_field(0)[0] == f1
    assert storage.extract_field(1)[0] == f2
    assert storage.extract_field(2)[0] == f3
    assert storage.extract_field(0)[0].label == "a"
    assert storage.extract_field(0, label="new label")[0].label == "new label"
    assert storage.extract_field(0)[0].label == "a"  # do not alter label
    assert storage.extract_field("a")[0] == f1
    assert storage.extract_field("b")[0] == f2
    assert storage.extract_field("c")[0] == f3
    with pytest.raises(ValueError):
        storage.extract_field("nonsense")


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES + [None])
def test_storage_apply(storage_factory):
    """test the apply function of StorageBase"""
    grid = UnitGrid([2])
    field = ScalarField(grid)

    s1 = MemoryStorage()
    s1.start_writing(field, info={"b": 2})
    field.data = np.array([0, 1])
    s1.append(field, 0)
    field.data = np.array([1, 2])
    s1.append(field, 1)
    s1.end_writing()

    out = None if storage_factory is None else storage_factory()
    s2 = s1.apply(lambda x: x + 1, out=out)
    assert storage_factory is None or s2 is out
    assert len(s2) == 2
    np.testing.assert_allclose(s2.times, s1.times)
    assert s2[0] == ScalarField(grid, [1, 2])
    assert s2[1] == ScalarField(grid, [2, 3])

    # test empty storage
    s1 = MemoryStorage()
    s2 = s1.apply(lambda x: x + 1)
    assert len(s2) == 0


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES + [None])
def test_storage_copy(storage_factory):
    """test the copy function of StorageBase"""
    grid = UnitGrid([2])
    field = ScalarField(grid)

    s1 = MemoryStorage()
    s1.start_writing(field, info={"b": 2})
    field.data = np.array([0, 1])
    s1.append(field, 0)
    field.data = np.array([1, 2])
    s1.append(field, 1)
    s1.end_writing()

    out = None if storage_factory is None else storage_factory()
    s2 = s1.copy(out=out)
    assert storage_factory is None or s2 is out
    assert len(s2) == 2
    np.testing.assert_allclose(s2.times, s1.times)
    assert s2[0] == s1[0]
    assert s2[1] == s1[1]

    # test empty storage
    s1 = MemoryStorage()
    s2 = s1.copy()
    assert len(s2) == 0


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES)
@pytest.mark.parametrize("dtype", [bool, complex])
def test_storage_types(storage_factory, dtype, rng):
    """test storing different types"""
    grid = UnitGrid([32])
    field = ScalarField.random_uniform(grid, rng=rng).copy(dtype=dtype)
    if dtype == complex:
        field += 1j * ScalarField.random_uniform(grid, rng=rng)

    s = storage_factory()
    s.start_writing(field)
    s.append(field, 0)
    s.append(field, 1)
    s.end_writing()

    assert len(s) == 2
    np.testing.assert_allclose(s.times, [0, 1])
    np.testing.assert_equal(s[0].data, field.data)
    np.testing.assert_equal(s[1].data, field.data)


@pytest.mark.multiprocessing
@pytest.mark.parametrize("storage_class", STORAGE_CLASSES)
def test_storage_mpi(storage_factory, rng):
    """test writing data using MPI"""
    eq = DiffusionPDE()
    grid = UnitGrid([8])
    field = ScalarField.random_normal(grid, rng=rng).smooth(1)

    storage = storage_factory()
    res = eq.solve(
        field, t_range=0.1, dt=0.001, backend="numpy", tracker=[storage.tracker(0.01)]
    )

    if mpi.is_main:
        assert res.integral == pytest.approx(field.integral)
        assert len(storage) == 11


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES)
def test_storing_transformation_collection(storage_factory, rng):
    """test transformation yielding field collections in storage classes"""
    grid = UnitGrid([8])
    field = ScalarField.random_normal(grid, rng=rng).smooth(1)

    def trans1(field, t):
        return FieldCollection([field, 2 * field + t])

    storage = storage_factory()
    eq = DiffusionPDE()
    trackers = [storage.tracker(0.01, transformation=trans1)]
    eq.solve(
        field,
        t_range=0.1,
        dt=0.001,
        backend="numpy",
        tracker=trackers,
    )

    assert storage.has_collection
    for t, sol in storage.items():
        a, a2 = sol
        np.testing.assert_allclose(a2.data, 2 * a.data + t)


@pytest.mark.parametrize("storage_class", STORAGE_CLASSES)
def test_storing_transformation_scalar(storage_factory, rng):
    """test transformations yielding scalar fields in storage classes"""
    grid = UnitGrid([8])
    field = ScalarField.random_normal(grid, rng=rng).smooth(1)

    storage = storage_factory()
    eq = DiffusionPDE(diffusivity=0)
    trackers = [storage.tracker(0.01, transformation=lambda f: f**2)]
    eq.solve(field, t_range=0.1, dt=0.001, backend="numpy", tracker=trackers)

    assert not storage.has_collection
    for sol in storage:
        np.testing.assert_allclose(sol.data, field.data**2)
