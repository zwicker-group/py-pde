'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import tempfile

import pytest
import numpy as np

from .. import FileStorage
from ...grids import UnitGrid
from ...grids.base import GridBase
from ...fields import ScalarField
from ...pdes import DiffusionPDE
from ...tools.misc import skipUnlessModule


        
@skipUnlessModule("h5py")
def test_storage_persistence():
    """ test writing to persistent trackers """
    dim = 5
    grid = UnitGrid([dim])
    field = ScalarField(grid)

    file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True)
    
    # write some data 
    for write_mode in ['append', 'truncate_once', 'truncate']:
        storage = FileStorage(file.name, info={'a': 1},
                              write_mode=write_mode)
        
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
        storage = FileStorage(file.name)
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
def test_simulation_persistence():
    """ test whether a tracker can accurately store information about
    simulation """
    file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True)
    storage = FileStorage(file.name)
     
    # write some simulation data
    pde = DiffusionPDE()
    grid = UnitGrid([16, 16])  # generate grid
    state = ScalarField.random_uniform(grid, 0.2, 0.3)
    pde.solve(state, t_range=0.11, dt=0.001,
              tracker=storage.tracker(interval=0.05))
    storage.close()        
     
    # read the data
    storage = FileStorage(file.name)
    np.testing.assert_almost_equal(storage.times, [0, 0.05, 0.1])
    data = np.array(storage.data)
    assert data.shape == (3,) + state.data.shape
    grid_res = GridBase.from_state(storage.info['grid'])
    assert grid == grid_res
    grid_res = storage.grid
    assert grid == grid_res



@skipUnlessModule("h5py")
def test_storage_fixed_size():
    """ test setting fixed size of FileStorage objects """
    c = ScalarField(UnitGrid([2]), data=1)

    for fixed in [True, False]:
        file = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True)
        storage = FileStorage(file.name, max_length=1 if fixed else None)
        assert len(storage) == 0
        
        storage.start_writing(c)
        assert len(storage) == 0
        storage.append(c.data, 0)
        assert len(storage) == 1
        
        if fixed:
            with pytest.raises(TypeError):
                storage.append(c.data, 1)
            assert len(storage) == 1
            np.testing.assert_allclose(storage.times, [0])
        else:
            storage.append(c.data, 1)
            assert len(storage) == 2
            np.testing.assert_allclose(storage.times, [0, 1])
            
            