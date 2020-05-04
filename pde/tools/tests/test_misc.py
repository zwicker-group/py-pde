'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import os
import json

import pytest
import numpy as np

from .. import misc



def test_ensure_directory_exists(tmp_path):
    """ tests the ensure_directory_exists function """
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
    
    
    
def test_environment():
    """ test the environment function """
    assert isinstance(misc.environment(), dict)
    
    
    
def test_preserve_scalars():
    """ test the preserve_scalars decorator """
    class Test():
        @misc.preserve_scalars
        def meth(self, arr):
            return arr + 1
        
    t = Test()
    
    assert t.meth(1) == 2
    np.testing.assert_equal(t.meth(np.ones(2)), np.full(2, 2))
    
    
    
def test_hybridmethod():
    """ test the hybridmethod decorator """
    class Test():
        @misc.hybridmethod
        def meth(cls):  # @NoSelf
            return 'class'
        
        @meth.instancemethod
        def meth(self):
            return 'instance'
        
    assert Test.meth() == 'class'
    assert Test().meth() == 'instance'
    
    
    
def test_estimate_computation_speed():    
    """ test estimate_computation_speed method """
    def f(x):
        return 2 * x
    
    def g(x):
        return np.sin(x) * np.cos(x)**2
    
    assert (misc.estimate_computation_speed(f, 1) >
            misc.estimate_computation_speed(g, 1))
    
    
    
def test_classproperty():
    """ test classproperty decorator """
    class Test():
        _value = 2
        
        @misc.classproperty
        def value(cls):  # @NoSelf
            return cls._value
        
    assert Test.value == 2
    
    
    
def test_progress_bars():
    """ test progress bars """
    for pb_cls in [misc.MockProgress, misc.get_progress_bar_class()]:
        tot = 0
        for i in pb_cls(range(4)):
            tot += i
        assert tot == 6
    
    
    
@misc.skipUnlessModule('h5py')
def test_hdf_write_attributes(tmp_path):
    """ test hdf_write_attributes function """
    import h5py
    path = tmp_path / "test_hdf_write_attributes.hdf5"

    # test normal case    
    data = {'a': 3, 'b': 'asd'}
    with h5py.File(path, 'w') as hdf_file:
        misc.hdf_write_attributes(hdf_file, data)
        data2 = {k: json.loads(v) for k, v in hdf_file.attrs.items()}
            
    assert data == data2
    assert data is not data2

    # test silencing of problematic items
    with h5py.File(path, 'w') as hdf_file:
        misc.hdf_write_attributes(hdf_file, {'a': 1, 'b': object()})
        data2 = {k: json.loads(v) for k, v in hdf_file.attrs.items()}
    assert data2 == {'a': 1}
            
    # test raising problematic items
    with h5py.File(path, 'w') as hdf_file:
        with pytest.raises(TypeError):
            misc.hdf_write_attributes(hdf_file, {'a': object()},
                                      raise_serialization_error=True)
                