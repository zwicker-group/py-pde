"""
Module containing functions for managing cache structures

.. autosummary::
   :nosignatures:

   cached_property
   cached_method
   hash_mutable
   hash_readable
   make_serializer
   make_unserializer
   DictFiniteCapacity
   SerializedDict


.. codeauthor:: David Zwicker <dzwicker@seas.harvard.edu>
"""

from __future__ import division

import collections  # @UnusedImport
import collections.abc
import functools
import logging
import numbers
from hashlib import sha1
from typing import Callable, Dict, Iterable, Optional

import numpy as np


def objects_equal(a, b) -> bool:
    """compares two objects to see whether they are equal

    In particular, this uses :func:`numpy.array_equal` to check for numpy arrays

    Args:
        a: The first object
        b: The second object

    Returns:
        bool: Whether the two objects are considered equal
    """
    # compare numpy arrays
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)  # type: ignore
    if isinstance(b, np.ndarray):
        return np.array_equal(b, a)  # type: ignore

    # compare dictionaries
    if isinstance(a, dict):
        if not isinstance(b, dict) or len(a) != len(b):
            return False
        return all(objects_equal(v, b[k]) for k, v in a.items())

    if isinstance(a, (tuple, list)):
        if a.__class__ != b.__class__ or len(a) != len(b):
            return False
        return all(objects_equal(x, y) for x, y in zip(a, b))

    # use direct comparison
    return a == b  # type: ignore


def _hash_iter(it: Iterable) -> int:
    """ get hash of an iterable but turning it into a tuple first """
    return hash(tuple(it))


def hash_mutable(obj) -> int:
    """return hash also for (nested) mutable objects. This function might be a
    bit slow, since it iterates over all containers and hashes objects
    recursively.

    Args:
        obj: A general python object

    Returns:
        int: A hash value associated with the data of `obj`
    """
    if hasattr(obj, "_cache_hash"):
        return int(obj._cache_hash())

    # deal with some special classes
    if isinstance(obj, (list, tuple)):
        return _hash_iter(hash_mutable(v) for v in obj)

    if isinstance(obj, (set, frozenset)):
        return hash(frozenset(hash_mutable(v) for v in obj))

    if isinstance(obj, collections.OrderedDict):
        return _hash_iter(
            (k, hash_mutable(v))
            for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_cache"))
        )

    unordered_mappings = (
        dict,
        collections.abc.MutableMapping,
        collections.defaultdict,
        collections.Counter,
    )
    if isinstance(obj, unordered_mappings):
        return hash(
            frozenset(
                (k, hash_mutable(v))
                for k, v in sorted(obj.items())
                if not (isinstance(k, str) and k.startswith("_cache"))
            )
        )

    if isinstance(obj, np.ndarray):
        return hash(obj.tobytes())

    try:
        # try using the internal hash function
        return hash(obj)
    except TypeError:
        try:
            # try hashing the data buffer
            return hash(sha1(obj))
        except (ValueError, TypeError):
            # otherwise, hash the internal dict
            return hash_mutable(obj.__dict__)


def hash_readable(obj) -> str:
    """return human readable hash also for (nested) mutable objects. This
    function returns a json-like representation of the object. The function
    might be a bit slow, since it iterates over all containers and hashes
    objects recursively. Note that this hash function tries to return the same
    value for equivalent objects, but it does not ensure that the objects can
    be reconstructed from this data.

    Args:
        obj: A general python object

    Returns:
        str: A hash value associated with the data of `obj`
    """
    if isinstance(obj, numbers.Number):
        return str(obj)

    if isinstance(obj, (str, bytes)):
        return '"' + str(obj).replace("\\", "\\\\").replace('"', '"') + '"'

    if isinstance(obj, (list, tuple)):
        return "[" + ", ".join(hash_readable(v) for v in obj) + "]"

    if isinstance(obj, (set, frozenset)):
        return "{" + ", ".join(hash_readable(v) for v in sorted(obj)) + "}"

    mappings = (
        dict,
        collections.abc.MutableMapping,
        collections.OrderedDict,
        collections.defaultdict,
        collections.Counter,
    )
    if isinstance(obj, mappings):
        hash_str = ", ".join(
            hash_readable(k) + ": " + hash_readable(v) for k, v in sorted(obj.items())
        )
        return "{" + hash_str + "}"

    if isinstance(obj, np.ndarray):
        return repr(obj)

    # otherwise, assume it's a generic object
    try:
        if hasattr(obj, "__getstate__"):
            data = obj.__getstate__()
        else:
            data = obj.__dict__

    except AttributeError:
        # strange object without a dictionary attached to it
        return repr(obj)

    else:
        # turn arguments into something readable
        args = ", ".join(
            str(k) + "=" + hash_readable(v)
            for k, v in sorted(data.items())
            if not k.startswith("_")
        )

        return "{name}({args})".format(name=obj.__class__.__name__, args=args)


def make_serializer(method: str) -> Callable:
    """returns a function that serialize data with the given method. Note that
    some of the methods destroy information and cannot be reverted.

    Args:
        method (str): An identifier determining the serializer that will be
            returned

    Returns:
        callable: A function that serializes objects
    """
    if callable(method):
        return method

    if method is None:
        return lambda s: s

    if method == "hash":
        return hash

    if method == "hash_mutable":
        return hash_mutable

    if method == "hash_readable":
        return hash_readable

    if method == "json":
        import json

        return lambda s: json.dumps(s, sort_keys=True).encode("utf-8")

    if method == "pickle":
        import pickle

        return lambda s: pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL)

    if method == "yaml":
        import yaml

        return lambda s: yaml.dump(s).encode("utf-8")

    raise ValueError("Unknown serialization method `%s`" % method)


def make_unserializer(method: str) -> Callable:
    """returns a function that unserialize data with the  given method

    This is the inverse function of :func:`make_serializer`.

    Args:
        method (str): An identifier determining the unserializer that will be
            returned

    Returns:
        callable: A function that serializes objects

    """
    if callable(method):
        return method

    if method is None:
        return lambda s: s

    if method == "json":
        import json

        return lambda s: json.loads(s.decode("utf-8"))

    if method == "pickle":
        import pickle

        return lambda s: pickle.loads(s)

    if method == "yaml":
        import yaml

        return yaml.full_load

    if method == "yaml_unsafe":
        import yaml  # @Reimport

        return yaml.unsafe_load

    raise ValueError("Unknown serialization method `%s`" % method)


class DictFiniteCapacity(collections.OrderedDict):
    """ cache with a limited number of items """

    default_capacity: int = 100

    def __init__(self, *args, **kwargs):
        self.capacity = kwargs.pop("capacity", self.default_capacity)
        super(DictFiniteCapacity, self).__init__(*args, **kwargs)

    def check_length(self):
        """ ensures that the dictionary does not grow beyond its capacity """
        while len(self) > self.capacity:
            self.popitem(last=False)

    def __eq__(self, other):
        return super().__eq__(other) and self.capacity == other.capacity

    def __ne__(self, other):
        return super().__ne__(other) or self.capacity != other.capacity

    def __setitem__(self, key, value):
        super(DictFiniteCapacity, self).__setitem__(key, value)
        self.check_length()

    def update(self, values):
        super(DictFiniteCapacity, self).update(values)
        self.check_length()


class SerializedDict(collections.abc.MutableMapping):
    """a key value database which is stored on the disk
    This class provides hooks for converting arbitrary keys and values to
    strings, which are then stored in the database.
    """

    def __init__(
        self,
        key_serialization: str = "pickle",
        value_serialization: str = "pickle",
        storage_dict: Optional[Dict] = None,
    ):
        """provides a dictionary whose keys and values are serialized

        Args:
            key_serialization (str):
                Determines the serialization method for keys
            value_serialization (str):
                Determines the serialization method for values
            storage_dict (dict):
                Can be used to chose a different dictionary for the underlying
                storage mechanism, e.g., storage_dict = PersistentDict()
        """
        # initialize the dictionary that actually stores the data
        if storage_dict is None:
            self._data = {}
        else:
            self._data = storage_dict

        # define the methods that serialize and unserialize the data
        self.serialize_key = make_serializer(key_serialization)
        self.unserialize_key = make_unserializer(key_serialization)
        self.serialize_value = make_serializer(value_serialization)
        self.unserialize_value = make_unserializer(value_serialization)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        # convert key to its string representation
        key_s = self.serialize_key(key)
        # fetch the value
        value = self._data[key_s]
        # convert the value to its object representation
        return self.unserialize_value(value)

    def __setitem__(self, key, value):
        # convert key and value to their string representations
        key_s = self.serialize_key(key)
        value_s = self.serialize_value(value)
        # add the item to the dictionary
        self._data[key_s] = value_s

    def __delitem__(self, key):
        # convert key to its string representation
        key_s = self.serialize_key(key)
        # delete the item from the dictionary
        del self._data[key_s]

    def __contains__(self, key):
        # convert key to its string representation
        key_s = self.serialize_key(key)
        # check whether this items exists in the dictionary
        return key_s in self._data

    def __iter__(self):
        # iterate  dictionary
        for key_s in self._data.__iter__():
            # convert the value to its object representation
            yield self.unserialize_key(key_s)


class _class_cache:
    """ class handling the caching of results of methods and properties """

    def __init__(
        self,
        factory=None,
        extra_args=None,
        ignore_args=None,
        hash_function="hash_mutable",
        doc=None,
        name=None,
    ):
        r"""decorator that caches calls in a dictionary attached to the
        instances. This can be used with most classes

        Example:
            An example for using the class is::

                class Foo():

                    @cached_property()
                    def property(self):
                        return "Cached property"

                    @cached_method()
                    def method(self):
                        return "Cached method"


                foo = Foo()
                foo.property
                foo.method()

        The cache can be cleared by setting `foo.\_cache\_methods = {}` if
        the cache factory is a simple dict, i.e, if `factory == None`.
        Alternatively, each cached method has a :func:`clear_cache_of_obj`
        method, which clears the cache of this particular method. In the example
        above we could thus call `foo.bar.clear\_cache\_of\_obj(foo)` to
        clear the cache.
        Note that the object instance has to be passed as a parameter, since the
        method :func:`bar` is defined on the class, not the instance, i.e., we
        could also call `Foo.bar.clear\_cache\_of\_obj(foo)`. To clear the
        cache from within a method, one can thus call
        `self.method_name.clear\_cache\_of\_obj(self)`, where
        `method\_name` is the name of the method whose cache is cleared

        Example:
            An advanced example is::

                class Foo():

                    def get_cache(self, name):
                        # `name` is the name of the method to cache
                        return DictFiniteCapacity()

                    @cached_method(factory='get_cache')
                    def foo(self):
                        return "Cached"

        Args:
            factory (callable):
                Function/class creating an empty cache. `dict` by default.
                This can be used with user-supplied storage backends by. The
                cache factory should return a dict-like object that handles the
                cache for the given method.
            extra_args (list):
                List of attributes of the class that are included in the cache
                key. They are then treated as if they are supplied as arguments
                to the method. This is important to include when the result of
                a method depends not only on method arguments but also on
                instance attributes.
            ignore_args (list):
                List of keyword arguments that are not included in the cache
                key. These should be arguments that do not influence the result
                of a method, e.g., because they only affect how intermediate
                results are displayed.
            hash_function (str):
                An identifier determining what hash function is used on the
                argument list.
            doc (str):
                Optional string giving the docstring of the decorated method
            name (str):
                Optional string giving the name of the decorated method
        """
        self.extra_args = extra_args
        self.hash_function = hash_function
        self.name = name

        # setup the ignored arguments
        if ignore_args is not None:
            if isinstance(ignore_args, str):
                ignore_args = [ignore_args]
            self.ignore_args = set(ignore_args)
        else:
            self.ignore_args = None

        # check whether the decorator has been applied correctly
        if callable(factory):
            class_name = self.__class__.__name__
            raise ValueError(
                f"Missing function call. Call this decorator as {class_name}() instead "
                f"of {class_name}"
            )

        else:
            self.factory = factory

    def _get_clear_cache_method(self) -> Callable:
        """return a method that can be attached to classes to clear the cache
        of the wrapped method"""

        def clear_cache(obj):
            """ clears the cache associated with this method """
            try:
                # try getting an initialized cache
                cache = obj._cache_methods[self.name]

            except (AttributeError, KeyError):
                # the cache was not initialized
                if self.factory is None:
                    # the cache would be a dictionary, but it is not yet
                    # initialized => we don't need to clear anything
                    return
                # initialize the cache, since it might open a persistent
                # database, which needs to be cleared
                cache = getattr(obj, self.factory)(self.name)

            # clear the cache
            cache.clear()

        return clear_cache

    def _get_wrapped_function(self, func: Callable) -> Callable:
        """ return the wrapped method, which implements the cache """

        if self.name is None:
            self.name = func.__name__

        # create the function to serialize the keys
        hash_key = make_serializer(self.hash_function)

        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            # try accessing the cache
            try:
                cache = obj._cache_methods[self.name]
            except (AttributeError, KeyError) as err:
                # the cache was not initialized
                wrapper._logger.debug("Initialize the cache for `%s`", self.name)
                if isinstance(err, AttributeError):
                    # the cache dictionary is not even present
                    obj._cache_methods = {}
                # create cache using the right factory method
                if self.factory is None:
                    cache = {}
                else:
                    cache = getattr(obj, self.factory)(self.name)
                # store the cache in the dictionary
                obj._cache_methods[self.name] = cache

            # determine the key that encodes the current arguments
            if self.ignore_args:
                kwargs_key = {
                    k: v for k, v in kwargs.items() if k not in self.ignore_args
                }
                func_args = [args, kwargs_key]
            else:
                func_args = [args, kwargs]

            if self.extra_args:
                for extra_arg in self.extra_args:
                    func_args.append(getattr(obj, extra_arg))

            cache_key = hash_key(tuple(func_args))

            try:
                # try loading the results from the cache
                result = cache[cache_key]
            except KeyError:
                # if this failed, compute and store the results
                wrapper._logger.debug(
                    "Cache missed. Compute result for method `%s` with args `%s`",
                    self.name,
                    func_args,
                )
                result = func(obj, *args, **kwargs)
                cache[cache_key] = result
            return result

        # initialize the logger
        wrapper._logger = logging.getLogger(__name__)  # type: ignore

        return wrapper


class cached_property(_class_cache):
    r"""Decorator to use a method as a cached property

    The function is only called the first time and each successive call returns
    the cached result of the first call.

    Example:
        Here is an example for how to use the decorator::

            class Foo():

                @cached_property
                def bar(self):
                    return "Cached"


            foo = Foo()
            result = foo.bar

    The data is stored in a dictionary named `_cache_methods` attached to
    the instance of each object. The cache can thus be cleared by setting
    `self.\_cache\_methods = {}`. The cache of specific property can be
    cleared using `self._cache_methods[property_name] = {}`, where
    `property\_name` is the name of the property

    Adapted from <https://wiki.python.org/moin/PythonDecoratorLibrary>.
    """

    def __call__(self, method: Callable):
        """ apply the cache decorator to the property """
        # save name, e.g., to be able to delete cache later
        self._cache_name = self.name
        self.clear_cache_of_obj = self._get_clear_cache_method()
        self.func = self._get_wrapped_function(method)

        self.__doc__ = self.func.__doc__
        self.__name__ = self.func.__name__
        self.__module__ = self.func.__module__
        return self

    def __get__(self, obj, owner):
        """ call the method to obtain the result for this property """
        return self.func(obj)


class cached_method(_class_cache):
    r"""Decorator to enable caching of a method

    The function is only called the first time and each successive call returns
    the cached result of the first call.

    Example:
        The decorator can be used like so::

            class Foo:

                @cached_method
                def bar(self):
                    return "Cached"


            foo = Foo()
            result = foo.bar()

    The data is stored in a dictionary named `\_cache\_methods` attached to
    the instance of each object. The cache can thus be cleared by setting
    `self.\_cache\_methods = {}`. The cache of specific property can be
    cleared using `self.\_cache\_methods[property\_name] = {}`, where
    `property\_name` is the name of the property
    """

    def __call__(self, method):
        """ apply the cache decorator to the method """

        wrapper = self._get_wrapped_function(method)

        # save name, e.g., to be able to delete cache later
        wrapper._cache_name = self.name
        wrapper.clear_cache_of_obj = self._get_clear_cache_method()

        return wrapper
