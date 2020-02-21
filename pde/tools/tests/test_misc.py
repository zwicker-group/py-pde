'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import os
import tempfile

import numpy as np

from .. import misc



def test_ensure_directory_exists():
    """ tests the ensure_directory_exists function """
    # create temporary name
    path = tempfile.mktemp()
    assert not os.path.exists(path)
    # create the folder
    misc.ensure_directory_exists(path)
    assert os.path.exists(path)
    # check that a second call has the same result
    misc.ensure_directory_exists(path)
    assert os.path.exists(path)
    # remove the folder again
    os.rmdir(path)
    assert not os.path.exists(path)
    
    
    
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
    
    