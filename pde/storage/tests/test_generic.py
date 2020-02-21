'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import functools
import tempfile

import numpy as np

from .. import MemoryStorage, FileStorage
from ...grids import UnitGrid
from ...fields import ScalarField
from ...pdes import DiffusionPDE
from ...tools.misc import module_available



def test_storage_write():
    """ test simple memory storage """
    dim = 5
    grid = UnitGrid([dim])
    field = ScalarField(grid)

    file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True)
    
    storage_classes = {'MemoryStorage': MemoryStorage}
    if module_available("h5py"):
        storage_classes['FileStorage'] = \
                                functools.partial(FileStorage, file.name)
    
    for name, storage_cls in storage_classes.items():
        storage = storage_cls(info={'a': 1})
        storage.start_writing(field, info={'b': 2})
        storage.append(np.arange(dim), 0)
        storage.append(np.arange(dim), 1)
        storage.end_writing()
            
        np.testing.assert_allclose(storage.times, np.arange(2))
        for f in storage:
            np.testing.assert_array_equal(f.data, np.arange(dim))
        for i in range(2):
            np.testing.assert_array_equal(storage[i].data, np.arange(dim))
        assert {'a': 1, 'b': 2}.items() <= storage.info.items()
        
        storage = storage_cls()
        storage.clear()
        for i in range(3):
            storage.start_writing(field)
            storage.append(np.arange(dim) + i, i)
            storage.end_writing()
         
        np.testing.assert_allclose(storage.times, np.arange(3),
                                   err_msg='storage class: ' + name)
        
        

def test_storage_truncation():
    """ test whether simple trackers can be used """
    for truncate in [True, False]:
        file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True)
        storages = [MemoryStorage()]
        if module_available("h5py"):
            storages.append(FileStorage(file.name))
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
            msg = f'truncate={truncate}, storage={storage}'
            np.testing.assert_allclose(storage.times, times, err_msg=msg) 
