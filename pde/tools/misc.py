"""
Miscallenous python functions 

.. autosummary::
   :nosignatures:

   module_available
   environment
   ensure_directory_exists
   preserve_scalars
   decorator_arguments
   skipUnlessModule
   import_class
   classproperty
   hybridmethod
   estimate_computation_speed
   hdf_write_attributes

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import errno
import functools
import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np

# import functions moved on 2020-07-27
# using this path for import is deprecated
from .output import display_progress, get_progress_bar_class  # @UnusedImport

Number = Union[float, complex]


def module_available(module_name: str) -> bool:
    """check whether a python module is available

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
    """obtain information about the compute environment

    Args:
        dict_type: The type to create the returned dictionaries. The default is
            `dict`, but :class:`collections.OrderedDict` is an alternative.

    Returns:
        dict: information about the python installation and packages
    """
    import matplotlib as mpl

    from .. import __version__ as package_version
    from .numba import numba_environment
    from .plotting import get_plotting_context

    def get_package_versions(packages: List[str]) -> Dict[str, str]:
        """ tries to load certain python packages and returns their version """
        versions: Dict[str, str] = dict_type()
        for name in sorted(packages):
            try:
                module = importlib.import_module(name)
            except ImportError:
                versions[name] = "not available"
            else:
                versions[name] = module.__version__  # type: ignore
        return versions

    result: Dict[str, Any] = dict_type()
    result["package version"] = package_version
    result["python version"] = sys.version
    result["platform"] = sys.platform

    # add details for mandatory packages
    result["mandatory packages"] = get_package_versions(
        ["matplotlib", "numba", "numpy", "scipy", "sympy"]
    )
    result["matplotlib environment"] = {
        "backend": mpl.get_backend(),
        "plotting context": get_plotting_context().__class__.__name__,
    }

    # add details about optional packages
    result["optional packages"] = get_package_versions(
        ["h5py", "pandas", "pyfftw", "tqdm"]
    )
    if module_available("numba"):
        result["numba environment"] = numba_environment()

    return result


def ensure_directory_exists(folder: Union[str, Path]):
    """creates a folder if it not already exists

    Args:
        folder (str): path of the new folder
    """
    folder = str(folder)
    if folder == "":
        return
    try:
        os.makedirs(folder)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


def preserve_scalars(method: Callable) -> Callable:
    """decorator that makes vectorized methods work with scalars

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
        args = [number_array(arg, copy=False) for arg in args]
        if args[0].ndim == 0:
            args = [arg[None] for arg in args]
            return method(self, *args)[0]
        else:
            return method(self, *args)

    return wrapper


def decorator_arguments(decorator: Callable) -> Callable:
    r"""make a decorator usable with and without arguments:

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


def skipUnlessModule(module_names: Union[Sequence[str], str]) -> Callable:
    """decorator that skips a test when a module is not available

    Args:
        module_names (str): The name of the required module(s)

    Returns:
        A function, so this can be used as a decorator
    """
    if isinstance(module_names, str):
        module_names = [module_names]

    for module_name in module_names:
        if not module_available(module_name):
            # return decorator skipping test
            return unittest.skip(f"requires {module_name}")

    # return no-op decorator if all modules are available
    def wrapper(f: Callable) -> Callable:
        return f

    return wrapper


def import_class(identifier: str):
    """import a class or module given an identifier

    Args:
        identifier (str):
            The identifier can be a module or a class. For instance, calling the
            function with the string `identifier == 'numpy.linalg.norm'` is
            roughly equivalent to running `from numpy.linalg import norm` and
            would return a reference to `norm`.
    """
    module_path, _, class_name = identifier.rpartition(".")
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # this happens when identifier does not contain a dot
        return importlib.import_module(class_name)


class classproperty(property):
    """decorator that can be used to define read-only properties for classes.

    This is inspired by the implementation of :mod:`astropy`, see
    `astropy.org <http://astropy.org/>`_.

    Example:
        The decorator can be used much like the `property` decorator::

            class Test():

                item: str = 'World'

                @classproperty
                def message(cls):
                    return 'Hello ' + cls.item

            print(Test.message)
    """

    def __new__(cls, fget=None, doc=None):
        if fget is None:
            # use wrapper to support decorator without arguments
            def wrapper(func):
                return cls(func)

            return wrapper

        return super().__new__(cls)

    def __init__(self, fget, doc=None):
        fget = self._wrap_fget(fget)

        super().__init__(fget=fget, doc=doc)

        if doc is not None:
            self.__doc__ = doc

    def __get__(self, obj, objtype):
        # The base property.__get__ will just return self here;
        # instead we pass objtype through to the original wrapped
        # function (which takes the class as its sole argument)
        return self.fget.__wrapped__(objtype)

    def getter(self, fget):
        return super().getter(self._wrap_fget(fget))

    def setter(self, fset):
        raise NotImplementedError("classproperty is read-only")

    def deleter(self, fdel):
        raise NotImplementedError("classproperty is read-only")

    @staticmethod
    def _wrap_fget(orig_fget):
        if isinstance(orig_fget, classmethod):
            orig_fget = orig_fget.__func__

        @functools.wraps(orig_fget)
        def fget(obj):
            return orig_fget(obj.__class__)

        return fget


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
        self.__isabstractmethod__ = bool(getattr(fclass, "__isabstractmethod__", False))

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
    """estimates the computation speed of a function

    Args:
        func (callable): The function to call

    Returns:
        float: the number of times the function can be calculated in one second.
        The inverse is thus the runtime in seconds per function call
    """
    import timeit

    test_duration = kwargs.pop("test_duration", 1)

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


def hdf_write_attributes(
    hdf_path, attributes: Dict[str, Any] = None, raise_serialization_error: bool = False
) -> None:
    """write (JSON-serialized) attributes to a hdf file

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


def number(value: Union[Number, str]) -> Number:
    """convert a value into a float or complex number

    Args:
        value (Number or str):
            The value which needs to be converted

    Result:
        Number: A complex number or a float if the imaginary part vanishes
    """
    result = complex(value)
    return result.real if result.imag == 0 else result


def get_common_dtype(*args):
    r"""returns a dtype in which all arguments can be represented

    Args:
        *args: All items (arrays, scalars, etc) to be checked

    Returns: np.complex if any entry is complex, otherwise np.double
    """
    for arg in args:
        if np.iscomplexobj(arg):
            return np.complex
    return np.double


def number_array(data: np.ndarray, dtype=None, copy: bool = True) -> np.ndarray:
    """convert array dtype either to np.double or np.complex

    Args:
        data (:class:`numpy.ndarray`):
            The data that needs to be converted to a float array. This can also be any
            iterable of numbers.
        dtype (numpy dtype):
            The data type of the field. All the numpy dtypes are supported. If omitted,
            it will be determined from `data` automatically.
        copy (bool):
            Whether the data must be copied (in which case the original array is left
            untouched). Note that data will always be copied when changing the dtype.

    Returns:
        :class:`numpy.ndarray`: An array with the correct dtype
    """
    if dtype is None:
        # dtype needs to be determined automatically
        try:
            # convert the result to a numpy array with the given dtype
            result = np.array(data, dtype=get_common_dtype(data), copy=copy)
        except TypeError:
            # Conversion can fail when `data` contains a complex sympy number, i.e.,
            # sympy.I. In this case, we simply try to convert the expression using a
            # complex dtype
            result = np.array(data, dtype=np.complex, copy=copy)

    else:
        # a specific dtype is requested
        result = np.array(data, dtype=np.dtype(dtype), copy=copy)

    return result
