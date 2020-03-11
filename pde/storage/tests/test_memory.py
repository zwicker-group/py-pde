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
    
    s3 = MemoryStorage.from_collection([s1, s2])
    np.testing.assert_allclose(np.ravel(s3.data), np.arange(4))



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

    