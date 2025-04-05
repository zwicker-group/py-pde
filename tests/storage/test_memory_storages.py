"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde import MemoryStorage, UnitGrid
from pde.fields import FieldCollection, ScalarField, Tensor2Field, VectorField


def test_memory_storage():
    """Test methods specific to memory storage."""
    sf = ScalarField(UnitGrid([1]))
    s1 = MemoryStorage()
    s1.start_writing(sf)
    sf.data = 0
    s1.append(sf, 0)
    sf.data = 2
    s1.append(sf, 1)

    s2 = MemoryStorage()
    s2.start_writing(sf)
    sf.data = 1
    s2.append(sf, 0)
    sf.data = 3
    s2.append(sf, 1)

    # test from_fields
    s3 = MemoryStorage.from_fields(s1.times, [s1[0], s1[1]])
    assert s3.times == s1.times
    np.testing.assert_allclose(s3.data, s1.data)

    # test from_collection
    s3 = MemoryStorage.from_collection([s1, s2])
    assert s3.times == s1.times
    np.testing.assert_allclose(np.ravel(s3.data), np.arange(4))


@pytest.mark.parametrize("cls", [ScalarField, VectorField, Tensor2Field])
def test_field_type_guessing_fields(cls, rng):
    """Test the ability to guess the field type."""
    grid = UnitGrid([3])
    field = cls.random_normal(grid, rng=rng)
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


def test_field_type_guessing_collection(rng):
    """Test the ability to guess the field type of a collection."""
    grid = UnitGrid([3])
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
