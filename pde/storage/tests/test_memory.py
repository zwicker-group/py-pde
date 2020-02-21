'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import numpy as np

from .. import MemoryStorage
from ...grids import UnitGrid
from ...fields import ScalarField


            
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
