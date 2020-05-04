'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np

from .. import FileStorage
from ...grids import UnitGrid
from ...fields import ScalarField
from ...pdes import DiffusionPDE
from ...tools.misc import skipUnlessModule


        
@skipUnlessModule("h5py")
@pytest.mark.parametrize('compression', [True, False])
def test_storage_persistence(compression, tmp_path):
    """ test writing to persistent trackers """
    dim = 5
    grid = UnitGrid([dim])
    field = ScalarField(grid)

    path = tmp_path / f'test_storage_persistence_{compression}.hdf5'
    
    # write some data 
    for write_mode in ['append', 'truncate_once', 'truncate']:
        storage = FileStorage(path, info={'a': 1},
                              write_mode=write_mode,
                              compression=compression)
        
        # first batch
        storage.start_writing(field, info={'b': 2})
        storage.append(np.arange(dim), 0)
        storage.append(np.arange(dim, 2*dim))
        storage.end_writing()
        
        # read first batch
        np.testing.assert_array_equal(storage.times, np.arange(2))
        np.testing.assert_array_equal(np.ravel(storage.data), np.arange(10))
        assert {'a': 1, 'b': 2}.items() <= storage.info.items()
        
        # second batch
        storage.start_writing(field, info={'c': 3})
        storage.append(np.arange(2*dim, 3*dim), 2)
        storage.end_writing()
        
        storage.close()
        
        # read the data
        storage = FileStorage(path)
        if write_mode == 'truncate':
            np.testing.assert_array_equal(storage.times, np.array([2]))
            np.testing.assert_array_equal(np.ravel(storage.data),
                                          np.arange(10, 15))
            assert storage.shape == (1, 5)
            info = {'c': 3}
            assert info.items() <= storage.info.items()
        else:
            np.testing.assert_array_equal(storage.times, np.arange(0, 3))
            np.testing.assert_array_equal(np.ravel(storage.data),
                                          np.arange(0, 15))
            assert storage.shape == (3, 5)
            info = {'a': 1, 'b': 2, 'c': 3}
            assert info.items() <= storage.info.items()
    
    
     
@skipUnlessModule("h5py")
@pytest.mark.parametrize('compression', [True, False])
def test_simulation_persistence(compression, tmp_path):
    """ test whether a tracker can accurately store information about
    simulation """
    path = tmp_path / 'test_simulation_persistence.hdf5'
    storage = FileStorage(path, compression=compression)
     
    # write some simulation data
    pde = DiffusionPDE()
    grid = UnitGrid([16, 16])  # generate grid
    state = ScalarField.random_uniform(grid, 0.2, 0.3)
    pde.solve(state, t_range=0.11, dt=0.001,
              tracker=storage.tracker(interval=0.05))
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
@pytest.mark.parametrize('compression', [True, False])
def test_storage_fixed_size(compression, tmp_path):
    """ test setting fixed size of FileStorage objects """
    c = ScalarField(UnitGrid([2]), data=1)

    for fixed in [True, False]:
        path = tmp_path / f'test_storage_fixed_size_{fixed}.hdf5'
        storage = FileStorage(path, max_length=1 if fixed else None,
                              compression=compression)
        assert len(storage) == 0
        
        storage.start_writing(c)
        assert len(storage) == 0
        storage.append(c.data, 0)
        assert len(storage) == 1
        
        if fixed:
            with pytest.raises((TypeError, ValueError)):
                storage.append(c.data, 1)
            assert len(storage) == 1
            np.testing.assert_allclose(storage.times, [0])
        else:
            storage.append(c.data, 1)
            assert len(storage) == 2
            np.testing.assert_allclose(storage.times, [0, 1])
            
            
            
@skipUnlessModule("h5py")
def test_appending(tmp_path):
    """ test the appending data """
    path = tmp_path / 'test_appending.hdf5'
    
    c = ScalarField(UnitGrid([2]), data=1)
    storage = FileStorage(path)
    storage.start_writing(c)
    assert len(storage) == 0
    storage.append(c.data, 0)
    assert storage._file_state == 'writing'
    assert len(storage) == 1
    storage.close()
                
    storage2 = FileStorage(path, write_mode='append')
    storage2.start_writing(c)
    storage2.append(c.data, 1)
    storage2.close()
    
    assert len(storage2) == 2
    

            
@skipUnlessModule("h5py")
def test_keep_opened(tmp_path):
    """ test the keep opened option """
    path = tmp_path / 'test_keep_opened.hdf5'
    
    c = ScalarField(UnitGrid([2]), data=1)
    storage = FileStorage(path, keep_opened=False)
    storage.start_writing(c)
    assert len(storage) == 0
    storage.append(c.data, 0)
    assert storage._file_state == 'closed'
    assert len(storage) == 1
    assert storage._file_state == 'reading'
    storage.append(c.data, 1)
    assert len(storage) == 2
                
    storage2 = FileStorage(path, write_mode='append')
    assert storage.times == storage2.times
    assert storage.data == storage2.data
    storage.close()  # close the old storage to enable writing here
    storage2.start_writing(c)
    storage2.append(c.data, 2)
    storage2.close()

    assert len(storage2) == 3
    np.testing.assert_allclose(storage2.times, np.arange(3))
    
                        