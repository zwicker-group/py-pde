'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from .. import MemoryStorage
from ...grids import UnitGrid
from ...fields import ScalarField, VectorField, Tensor2Field, FieldCollection


            
def test_memory_storage():
    """ test methods specific to memory storage """
    sf = ScalarField(UnitGrid([1]))
    s1 = MemoryStorage()
    s1.start_writing(sf)
    s1.append(np.array([0]), 0)
    s1.append(np.array([2]), 1)
    s2 = MemoryStorage()
    s2.start_writing(sf)
    s2.append(np.array([1]), 0)
    s2.append(np.array([3]), 1)

    # test from_fields
    s3 = MemoryStorage.from_fields(s1.times, [s1[0], s1[1]])
    assert s3.times == s1.times
    np.testing.assert_allclose(s3.data, s1.data)
    
    # test from_collection
    s3 = MemoryStorage.from_collection([s1, s2])
    assert s3.times == s1.times
    np.testing.assert_allclose(np.ravel(s3.data), np.arange(4))
    
    # test extraction
    s4 = s1.extract_time_range()
    assert s4.times == s1.times
    np.testing.assert_allclose(s4.data, s1.data)
    s4 = s1.extract_time_range(0.5)
    assert s4.times == s1.times[:1]
    np.testing.assert_allclose(s4.data, s1.data[:1])
    s4 = s1.extract_time_range((0.5, 1.5))
    assert s4.times == s1.times[1:]
    np.testing.assert_allclose(s4.data, s1.data[1:])



def test_memory_storage_collection():
    """ test methods specific to FieldCollections in memory storage """
    grid = UnitGrid([2, 2])
    f1 = ScalarField.random_uniform(grid, 0.1, 0.4)
    f2 = VectorField.random_uniform(grid, 0.1, 0.4)
    f3 = Tensor2Field.random_uniform(grid, 0.1, 0.4)
    fc = FieldCollection([f1, f2, f3])    
    
    # store some data
    storage = MemoryStorage()
    storage.start_writing(fc)
    storage.append(fc.data, 0)
    storage.append(fc.data, 1)
    storage.end_writing()
    
    assert storage.extract_field(0)[0] == f1
    assert storage.extract_field(1)[0] == f2
    assert storage.extract_field(2)[0] == f3



def test_field_type_guessing():
    """ test the ability to guess the field type """
    for cls in [ScalarField, VectorField, Tensor2Field]:
        grid = UnitGrid([3])
        field = cls.random_normal(grid)
        s = MemoryStorage()
        s.start_writing(field)
        s.append(field.data, 0)
        s.append(field.data, 1)
        
        # delete information
        s._field = None
        s.info = {}
        
        assert len(s) == 2
        assert s[0] == field
        
    field = FieldCollection([ScalarField(grid), VectorField(grid)])
    s = MemoryStorage()
    s.start_writing(field)
    s.append(field.data, 0)
    
    # delete information
    s._field = None
    s.info = {}
    
    with pytest.raises(RuntimeError):
        s[0]

    