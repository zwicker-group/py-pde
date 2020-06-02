'''
Miscallenous python functions 

.. autosummary::
   :nosignatures:

   module_available
   environment
   ensure_directory_exists
   preserve_scalars
   decorator_arguments
   skipUnlessModule
   get_progress_bar_class
   display_progress
   add_scaled_colorbar
   import_class
   classproperty
   hybridmethod
   estimate_computation_speed
   hdf_write_attributes
   in_jupyter_notebook

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import os
import errno
import functools
import json
import sys
import importlib
import unittest
import warnings
from pathlib import Path
from typing import Callable, Dict, Any, List, Union

import numpy as np



def module_available(module_name: str) -> bool:
    """ check whether a python module is available
    
    Args:
        module_name (str): The name of the module
        
    Returns:
        `True` if the module can be imported and `False` otherwise
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        return False
    else:
        return True



def environment(dict_type=dict) -> Dict[str, Any]:
    """ obtain information about the compute environment
    
    Args:
        dict_type: The type to create the returned dictionaries. The default is
            `dict`, but :class:`collections.OrderedDict` is an alternative.
    
    Returns:
        dict: information about the python installation and packages
    """
    from .. import __version__ as package_version

    from .numba import numba_environment

    def get_package_versions(packages: List[str]) -> Dict[str, str]:
        """ tries to load certain python packages and returns their version """
        versions: Dict[str, str] = dict_type()
        for name in sorted(packages):
            try:
                module = importlib.import_module(name)
            except ImportError:
                versions[name] = 'not available'
            else:
                versions[name] = module.__version__  # type: ignore
        return versions

    result: Dict[str, Any] = dict_type()
    result['package version'] = package_version
    result['python version'] = sys.version
    result['mandatory packages'] = get_package_versions(
                            ['matplotlib', 'numba', 'numpy', 'scipy', 'sympy'])
    result['optional packages'] = get_package_versions(['h5py', 'pandas',
                                                        'pyfftw', 'tqdm'])
    if module_available('numba'):
        result['numba environment'] = numba_environment()
    
    return result
    
    
    

def ensure_directory_exists(folder: Union[str, Path]):
    """ creates a folder if it not already exists
    
    Args:
        folder (str): path of the new folder
    """
    folder = str(folder)
    if folder == '':
        return
    try:
        os.makedirs(folder)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
    


def preserve_scalars(method: Callable) -> Callable:
    """ decorator that makes vectorized methods work with scalars
    
    This decorator allows to call functions that are written to work on numpy
    arrays to also accept python scalars, like `int` and `float`. Essentially,
    this wrapper turns them into an array and unboxes the result.

    Args:
        method: The method being decorated
        
    Returns:
        The decorated method
    """
    @functools.wraps(method)
    def wrapper(self, *args):
        args = [np.asanyarray(arg, dtype=np.double)
                for arg in args]
        if args[0].ndim == 0:
            args = [arg[None] for arg in args]
            return method(self, *args)[0]
        else:
            return method(self, *args)
    return wrapper



def decorator_arguments(decorator: Callable) -> Callable:
    r""" make a decorator usable with and without arguments:
    
    The resulting decorator can be used like `@decorator`
    or `@decorator(\*args, \**kwargs)`
    
    Inspired by https://stackoverflow.com/a/14412901/932593
    
    Args:
        decorator: the decorator that needs to be modified
        
    Returns:
        the decorated function
    """
    @functools.wraps(decorator)
    def new_decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return decorator(args[0])
        else:
            # decorator arguments
            return lambda realf: decorator(realf, *args, **kwargs)

    return new_decorator



def skipUnlessModule(module_name: str) -> Callable:
    """ decorator that skips a test when a module is not available
    
    Args:
        module_name (str): The name of the required module
        
    Returns:
        A function, so this can be used as a decorator
    """
    if module_available(module_name):
        # return no-op decorator
        def wrapper(f: Callable) -> Callable:
            return f
        return wrapper
    else:
        # return decorator skipping test
        return unittest.skip(f"requires {module_name}")
        


class MockProgress():
    """ indicates progress by printing dots to stderr """
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.n = self.total = 0
        self.disable = False

    def __iter__(self):
        return iter(self.iterable)

    def close(self, *args, **kwargs): pass
    
    def refresh(self, *args, **kwargs): 
        sys.stderr.write('.')
        sys.stderr.flush()
        
    def set_description(self, msg: str, refresh: bool = True, *args,
                        **kwargs):
        if refresh:
            self.refresh()



def get_progress_bar_class():
    """ returns a class that behaves as progress bar.
    
    This either uses classes from the optional `tqdm` package or a simple
    version that writes dots to stderr, if the class it not available.
    """
    try:
        # try importing the tqdm package
        import tqdm
        
    except ImportError:
        # create a mock class, since tqdm is not available 
        # progress bar package does not seem to be available
        warnings.warn('`tqdm` package is not available. Progress will '
                      'be indicated by dots.')
        progress_bar_class = MockProgress

    else:
        # tqdm is available => decide which class to return   
        tqdm_version = tuple(int(v) for v in tqdm.__version__.split('.')[:2])
        if tqdm_version >= (4, 40):
            # optionally import notebook progress bar in recent version
            try:
                # check whether progress bar can use a widget
                import ipywidgets  # @UnusedImport  
            except ImportError:
                # widgets are not available => use standard tqdm
                progress_bar_class = tqdm.tqdm
            else:
                # use the fancier version of the progress bar in jupyter
                from tqdm.auto import tqdm as progress_bar_class
        else:
            # only import text progress bar in older version
            progress_bar_class = tqdm.tqdm
            warnings.warn('Your version of tqdm is outdated. To get a nicer '
                          'progress bar update to at least version 4.40.')

    return progress_bar_class
        
        

def display_progress(iterator, total=None, enabled=True, **kwargs):
    r"""
    displays a progress bar when iterating
    
    Args:
        iterator (iter): The iterator
        total (int): Total number of steps
        enabled (bool): Flag determining whether the progress is display
        **kwargs: All extra arguments are forwarded to the progress bar class
        
    Returns:
        A class that behaves as the original iterator, but shows the progress
        alongside iteration.
    """
    if not enabled:
        return iterator
    
    return get_progress_bar_class()(iterator, total=total, **kwargs)



def add_scaled_colorbar(im,
                        ax=None,
                        aspect: float = 20,
                        pad_fraction: float = 0.5,
                        **kwargs):
    """ add a vertical color bar to an image plot
    
    The height of the colorbar is now adjusted to the plot, so that the width
    determined by `aspect` is now given relative to the height. Moreover, the
    gap between the colorbar and the plot is now given in units of the fraction
    of the width by `pad_fraction`. 

    Inspired by https://stackoverflow.com/a/33505522/932593
    
    Args:
        im: object returned from :meth:`matplotlib.pyplot.imshow`
        ax (:class:`matplotlib.axes.Axes`): the current figure axes
        aspect (float): the target aspect ratio of the colorbar
        pad_fraction (float): Width of the gap between colorbar and image
        **kwargs: Additional parameters are passed to colorbar call

    Returns:
        the result of the colorbar call
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    if ax is not None:
        plt.sca(ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)            
       


def import_class(identifier: str):
    """ import a class or module given an identifier 
    
    Args:
        identifier (str):
            The identifier can be a module or a class. For instance, calling the
            function with the string `identifier == 'numpy.linalg.norm'` is
            roughly equivalent to running `from numpy.linalg import norm` and
            would return a reference to `norm`.
    """
    module_path, _, class_name = identifier.rpartition('.')
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # this happens when identifier does not contain a dot
        return importlib.import_module(class_name)
    


class classproperty:
    """ decorator that can be used to define read-only properties for classes. 
    Adopted from http://stackoverflow.com/a/5192374/932593
    """
    def __init__(self, f):
        self.f = f
         
    def __get__(self, obj, owner):
        return self.f(owner)
    


class hybridmethod:
    """
    descriptor that can be used as a decorator to allow calling a method both
    as a classmethod and an instance method
     
    Adapted from https://stackoverflow.com/a/28238047
    """
    
    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = doc or fclass.__doc__
        # support use on abstract base classes
        self.__isabstractmethod__ = bool(
            getattr(fclass, '__isabstractmethod__', False)
        )
        

    def classmethod(self, fclass):
        return type(self)(fclass, self.finstance, None)


    def instancemethod(self, finstance):
        return type(self)(self.fclass, finstance, self.__doc__)


    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            # either bound to the class, or no instance method available
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)    
    
    
    
def estimate_computation_speed(func: Callable, *args, **kwargs) -> float:
    """ estimates the computation speed of a function
    
    Args:
        func (callable): The function to call
    
    Returns:
        float: the number of times the function can be calculated in one second.
        The inverse is thus the runtime in seconds per function call
    """
    import timeit
    test_duration = kwargs.pop('test_duration', 1)
    
    # prepare the function
    if args or kwargs:
        test_func = functools.partial(func, *args, **kwargs)
    else:
        test_func = func  # type: ignore
    
    # call function once to allow caches be filled
    test_func()
     
    # call the function until the total time is achieved
    number, duration = 1, 0
    while duration < 0.1 * test_duration:
        number *= 10
        duration = timeit.timeit(test_func, number=number)  # type: ignore
    return number / duration



def hdf_write_attributes(hdf_path,
                         attributes: Dict[str, Any] = None,
                         raise_serialization_error: bool = False) -> None:
    """ write (JSON-serialized) attributes to a hdf file
    
    Args:
        hdf_path:
            Path to a group or dataset in an open HDF file
        attributes (dict):
            Dictionary with values written as attributes
        raise_serialization_error (bool):
            Flag indicating whether serialization errors are raised or silently
            ignored
    """ 
    if attributes is None:
        return
        
    for key, value in attributes.items():
        try:
            value_serialized = json.dumps(value)
        except TypeError:
            if raise_serialization_error:
                raise
        else:
            hdf_path.attrs[key] = value_serialized



def in_jupyter_notebook() -> bool:
    """ checks whether we are in a jupyter notebook """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
        
    try:
        ipython_config = get_ipython().config
    except AttributeError:
        return False
        
    return ('IPKernelApp' in ipython_config)
