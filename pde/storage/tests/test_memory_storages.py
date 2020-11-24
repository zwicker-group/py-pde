"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from pde import MemoryStorage, UnitGrid
from pde.fields import FieldCollection, ScalarField, Tensor2Field, VectorField


def test_memory_storage():
    """ test methods specific to memory storage """
    sf = ScalarField(UnitGrid([1]))
    s1 = MemoryStorage()
    s1.start_writing(sf)
    s1.append(sf.copy(data=0), 0)
    s1.append(sf.copy(data=2), 1)
    s2 = MemoryStorage()
    s2.start_writing(sf)
    s2.append(sf.copy(data=1), 0)
    s2.append(sf.copy(data=3), 1)

    # test from_fields
    s3 = MemoryStorage.from_fields(s1.times, [s1[0], s1[1]])
    assert s3.times == s1.times
    np.testing.assert_allclose(s3.data, s1.data)

    # test from_collection
    s3 = MemoryStorage.from_collection([s1, s2])
    assert s3.times == s1.times
    np.testing.assert_allclose(np.ravel(s3.data), np.arange(4))


def test_field_type_guessing():
    """ test the ability to guess the field type """
    for cls in [ScalarField, VectorField, Tensor2Field]:
        grid = UnitGrid([3])
        field = cls.random_normal(grid)
        s = MemoryStorage()
        s.start_writing(field)
        s.append(field, 0)
        s.append(field, 1)

        # delete information
        s._field = None
        s.info = {}

        assert not s.has_collection
        assert len(s) == 2
        assert s[0] == field

    field = FieldCollection([ScalarField(grid), VectorField(grid)])
    s = MemoryStorage()
    s.start_writing(field)
    s.append(field, 0)

    assert s.has_collection

    # delete information
    s._field = None
    s.info = {}

    with pytest.raises(RuntimeError):
        s[0]
