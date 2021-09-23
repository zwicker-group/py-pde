"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
import os

import numpy as np
import pytest

from pde.tools import misc


def test_ensure_directory_exists(tmp_path):
    """tests the ensure_directory_exists function"""
    # create temporary name
    path = tmp_path / "test_ensure_directory_exists"
    assert not path.exists()
    # create the folder
    misc.ensure_directory_exists(path)
    assert path.is_dir()
    # check that a second call has the same result
    misc.ensure_directory_exists(path)
    assert path.is_dir()
    # remove the folder again
    os.rmdir(path)
    assert not path.exists()


def test_preserve_scalars():
    """test the preserve_scalars decorator"""

    class Test:
        @misc.preserve_scalars
        def meth(self, arr):
            return arr + 1

    t = Test()

    assert t.meth(1) == 2
    np.testing.assert_equal(t.meth(np.ones(2)), np.full(2, 2))


def test_hybridmethod():
    """test the hybridmethod decorator"""

    class Test:
        @misc.hybridmethod
        def method(cls):  # @NoSelf
            return "class"

        @method.instancemethod
        def method(self):
            return "instance"

    assert Test.method() == "class"
    assert Test().method() == "instance"


def test_estimate_computation_speed():
    """test estimate_computation_speed method"""

    def f(x):
        return 2 * x

    def g(x):
        return np.sin(x) * np.cos(x) ** 2

    assert misc.estimate_computation_speed(f, 1) > misc.estimate_computation_speed(g, 1)


def test_classproperty():
    """test classproperty decorator"""

    class Test:
        _value = 2

        @misc.classproperty
        def value(cls):  # @NoSelf
            return cls._value

    assert Test.value == 2


@misc.skipUnlessModule("h5py")
def test_hdf_write_attributes(tmp_path):
    """test hdf_write_attributes function"""
    import h5py

    path = tmp_path / "test_hdf_write_attributes.hdf5"

    # test normal case
    data = {"a": 3, "b": "asd"}
    with h5py.File(path, "w") as hdf_file:
        misc.hdf_write_attributes(hdf_file, data)
        data2 = {k: json.loads(v) for k, v in hdf_file.attrs.items()}

    assert data == data2
    assert data is not data2

    # test silencing of problematic items
    with h5py.File(path, "w") as hdf_file:
        misc.hdf_write_attributes(hdf_file, {"a": 1, "b": object()})
        data2 = {k: json.loads(v) for k, v in hdf_file.attrs.items()}
    assert data2 == {"a": 1}

    # test raising problematic items
    with h5py.File(path, "w") as hdf_file:
        with pytest.raises(TypeError):
            misc.hdf_write_attributes(
                hdf_file, {"a": object()}, raise_serialization_error=True
            )
