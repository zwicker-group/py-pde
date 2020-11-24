"""
.. codeauthor:: David Zwicker <dzwicker@seas.harvard.edu>
"""

from __future__ import division

import collections
import copy
import sys

import numpy as np
import pytest
from pde.tools import cache


def deep_getsizeof(obj, ids=None):
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    Function modified from
    https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609
    """
    if ids is not None:
        if id(obj) in ids:
            return 0
    else:
        ids = set()

    r = sys.getsizeof(obj)
    ids.add(id(obj))

    if isinstance(obj, str):
        # simple string
        return r

    if isinstance(obj, collections.abc.Mapping):
        # simple mapping
        return r + sum(
            deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in obj.items()
        )

    if isinstance(obj, collections.abc.Container):
        # collection that is neither a string nor a mapping
        return r + sum(deep_getsizeof(x, ids) for x in obj)

    if hasattr(obj, "__dict__"):
        # custom object
        return r + deep_getsizeof(obj.__dict__, ids)

    # basic object: neither of the above
    return r


def test_objects_equal():
    """ test the objects_equal function """
    # basic python objects
    eq = cache.objects_equal
    assert eq(1, 1)
    assert eq(1, 1.0)
    assert eq((1,), (1,))
    assert eq([1], [1])
    assert eq({"a": 1}, {"a": 1})

    assert not eq(1, "1")
    assert not eq(1, (1,))
    assert not eq(1, [1])
    assert not eq((1,), [1])
    assert not eq(1, {"a": 1})
    assert not eq({"a": 1}, {"a": 1, "b": 2})

    # test numpy arrays
    a = np.arange(3)
    b = np.arange(4)
    assert eq(a, a)
    assert eq(a, [0, 1, 2])
    assert eq([0, 1, 2], a)
    assert not eq(a, b)
    assert eq([a], [a])
    assert eq({"a": a}, {"a": a})


def get_serialization_methods(with_none=True):
    """ returns possible methods for serialization that are supported """
    methods = ["json", "pickle"]

    if with_none:
        methods.append(None)

    # check whether yaml is actually available
    try:
        import yaml  # @UnusedImport
    except ImportError:
        pass
    else:
        methods.append("yaml")

    return methods


def test_hashes():
    """ test whether the hash key makes sense """

    class Dummy:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return self.value

    for f in (cache.hash_mutable, cache.hash_readable):
        # test simple objects
        for obj in (
            1,
            1.2,
            "a",
            (1, 2),
            [1, 2],
            {1, 2},
            {1: 2},
            {(1, 2): [2, 3], (1, 3): [1, 2]},
            Dummy(1),
            np.arange(5),
        ):
            o2 = copy.deepcopy(obj)
            assert f(obj) == f(o2)

        # make sure different objects get different hash
        assert f(1) != f("1")
        assert f("a") != f("b")
        assert f({1, 2}) != f((1, 2))


def test_serializer_nonsense():
    """ test whether errors are thrown for wrong input """
    with pytest.raises(ValueError):
        cache.make_serializer("non-sense")
    with pytest.raises(ValueError):
        cache.make_unserializer("non-sense")


@pytest.mark.parametrize("method", get_serialization_methods())
def test_serializer(method):
    """ tests whether the make_serializer returns a canonical hash """
    encode = cache.make_serializer(method)

    assert encode(1) == encode(1)

    assert encode([1, 2, 3]) != encode([2, 3, 1])
    if method != "json":
        # json cannot encode sets
        assert encode({1, 2, 3}) == encode({2, 3, 1})


def test_serializer_hash_mutable():
    """ tests whether the make_serializer returns a canonical hash """
    # test special serializer
    encode = cache.make_serializer("hash_mutable")
    assert encode({"a": 1, "b": 2}) == encode({"b": 2, "a": 1})

    c1 = collections.OrderedDict([("a", 1), ("b", 2)])
    c2 = collections.OrderedDict([("b", 2), ("a", 1)])
    assert cache.hash_mutable(c1) != cache.hash_mutable(c2)
    assert cache.hash_mutable(dict(c1)) == cache.hash_mutable(dict(c2))

    class Test:
        """ test class that neither implements __eq__ nor __hash__ """

        def __init__(self, a):
            self.a = a

    assert cache.hash_mutable(Test(1)) != cache.hash_mutable(Test(1))

    class TestEq:
        """ test class that only implements __eq__ and not __hash__ """

        def __init__(self, a):
            self.a = a

        def __eq__(self, other):
            return self.a == other.a

    assert cache.hash_mutable(TestEq(1)) == cache.hash_mutable(TestEq(1))
    assert cache.hash_mutable(TestEq(1)) != cache.hash_mutable(TestEq(2))


def test_unserializer():
    """tests whether the make_serializer and make_unserializer return the
    original objects"""
    data_list = [None, 1, [1, 2], {"b": 1, "a": 2}]
    for method in get_serialization_methods():
        encode = cache.make_serializer(method)
        decode = cache.make_unserializer(method)
        for data in data_list:
            assert data == decode(encode(data))


def _test_SerializedDict(
    storage, reinitialize=None, key_serialization="pickle", value_serialization="pickle"
):
    """ tests the SerializedDict class with a particular parameter set """
    data = cache.SerializedDict(
        key_serialization, value_serialization, storage_dict=storage
    )

    if value_serialization == "none":
        with pytest.raises(TypeError):
            data["a"] = 1

        v1, v2, v3 = "1", "2", "3"

    else:
        v1, v2, v3 = 1, 2, "3"

    data["a"] = v1
    assert len(data) == v1
    data["b"] = v2
    assert data["b"] == v2

    assert len(data) == v2
    del data["a"]
    assert len(data) == v1
    with pytest.raises(KeyError):
        data["a"]

    data.update({"d": v3})
    assert len(data) == v2

    # reinitialize the storage dictionary
    if reinitialize is not None:
        data._data = reinitialize()
    assert len(data) == v2
    assert data["b"] == v2
    assert "d" in data
    assert {"b", "d"} == set(data.keys())
    assert {v2, v3} == set(data.values())
    data.clear()
    assert len(data) == 0

    # reinitialize the dictionary
    if reinitialize is not None:
        data._data = reinitialize()
    assert len(data) == 0


@pytest.mark.parametrize("cache_storage", [None, "get_finite_dict"])
def test_property_cache(cache_storage):
    """ test cached_property decorator """

    # create test class
    class CacheTest:
        """ class for testing caching """

        def __init__(self):
            self.counter = 0

        def get_finite_dict(self, n):
            return cache.DictFiniteCapacity(capacity=1)

        @property
        def uncached(self):
            self.counter += 1
            return 1

        def cached(self):
            self.counter += 1
            return 2

    # apply the cache with the given storage
    if cache_storage is None:
        decorator = cache.cached_property()
    else:
        decorator = cache.cached_property(cache_storage)
    CacheTest.cached = decorator(CacheTest.cached)

    # try to objects to make sure caching is done on the instance level
    for obj in [CacheTest(), CacheTest()]:
        # test uncached method
        assert obj.uncached == 1
        assert obj.counter == 1
        assert obj.uncached == 1
        assert obj.counter == 2
        obj.counter = 0

        # test cached methods
        assert obj.cached == 2
        assert obj.counter == 1
        assert obj.cached == 2
        assert obj.counter == 1


@pytest.mark.parametrize("serializer", get_serialization_methods(with_none=False))
@pytest.mark.parametrize("cache_factory", [None, "get_finite_dict"])
def test_method_cache(serializer, cache_factory):
    """ test one particular parameter set of the cached_method decorator """

    # create test class
    class CacheTest:
        """ class for testing caching """

        def __init__(self):
            self.counter = 0

        def get_finite_dict(self, name):
            return cache.DictFiniteCapacity(capacity=1)

        def uncached(self, arg):
            self.counter += 1
            return arg

        @cache.cached_method(hash_function=serializer, factory=cache_factory)
        def cached(self, arg):
            self.counter += 1
            return arg

        @cache.cached_method(hash_function=serializer, factory=cache_factory)
        def cached_kwarg(self, a=0, b=0):
            self.counter += 1
            return a + b

    # test what happens when the decorator is applied wrongly
    with pytest.raises(ValueError):
        cache.cached_method(CacheTest.cached)

    # try to objects to make sure caching is done on the instance level and
    # that clearing the cache works
    obj1, obj2 = CacheTest(), CacheTest()
    for k, obj in enumerate([obj1, obj2, obj1]):

        # clear the cache before the first and the last pass
        if k == 0 or k == 2:
            CacheTest.cached.clear_cache_of_obj(obj)
            CacheTest.cached_kwarg.clear_cache_of_obj(obj)
            obj.counter = 0

        # test uncached method
        assert obj.uncached(1) == 1
        assert obj.counter == 1
        assert obj.uncached(1) == 1
        assert obj.counter == 2
        obj.counter = 0

        # test cached methods
        for method in (obj.cached, obj.cached_kwarg):
            # run twice to test clearing the cache
            for _ in (None, None):
                # test simple caching behavior
                assert method(1) == 1
                assert obj.counter == 1
                assert method(1) == 1
                assert obj.counter == 1
                assert method(2) == 2
                assert obj.counter == 2
                assert method(2) == 2
                assert obj.counter == 2

                # test special properties of cache_factories
                if cache_factory is None:
                    assert method(1) == 1
                    assert obj.counter == 2
                elif cache_factory == "get_finite_dict":
                    assert method(1) == 1
                    assert obj.counter == 3
                else:
                    raise ValueError("Unknown cache_factory `%s`" % cache_factory)

                obj.counter = 0
                # clear cache to test the second run
                method.clear_cache_of_obj(obj)

        # test complex cached method
        assert obj.cached_kwarg(1, b=2) == 3
        assert obj.counter == 1
        assert obj.cached_kwarg(1, b=2) == 3
        assert obj.counter == 1
        assert obj.cached_kwarg(2, b=2) == 4
        assert obj.counter == 2
        assert obj.cached_kwarg(2, b=2) == 4
        assert obj.counter == 2
        assert obj.cached_kwarg(1, b=3) == 4
        assert obj.counter == 3
        assert obj.cached_kwarg(1, b=3) == 4
        assert obj.counter == 3


@pytest.mark.parametrize("serializer", get_serialization_methods(with_none=False))
@pytest.mark.parametrize("cache_factory", [None, "get_finite_dict"])
def test_method_cache_extra_args(serializer, cache_factory):
    """ test extra arguments in the cached_method decorator """
    # create test class
    class CacheTest:
        """ class for testing caching """

        def __init__(self, value=0):
            self.counter = 0
            self.value = 0

        def get_finite_dict(self, name):
            return cache.DictFiniteCapacity(capacity=1)

        @cache.cached_method(
            hash_function=serializer, extra_args=["value"], factory=cache_factory
        )
        def cached(self, arg):
            self.counter += 1
            return self.value + arg

    obj = CacheTest(0)

    # test simple caching behavior
    assert obj.cached(1) == 1
    assert obj.counter == 1
    assert obj.cached(1) == 1
    assert obj.counter == 1
    assert obj.cached(2) == 2
    assert obj.counter == 2
    assert obj.cached(2) == 2
    assert obj.counter == 2

    obj.value = 10
    # test simple caching behavior
    assert obj.cached(1) == 11
    assert obj.counter == 3
    assert obj.cached(1) == 11
    assert obj.counter == 3
    assert obj.cached(2) == 12
    assert obj.counter == 4
    assert obj.cached(2) == 12
    assert obj.counter == 4


@pytest.mark.parametrize("serializer", get_serialization_methods(with_none=False))
@pytest.mark.parametrize("cache_factory", [None, "get_finite_dict"])
@pytest.mark.parametrize("ignore_args", ["display", ["display"]])
def test_method_cache_ignore(serializer, cache_factory, ignore_args):
    """ test ignored parameters of the cached_method decorator """
    # create test class
    class CacheTest:
        """ class for testing caching """

        def __init__(self):
            self.counter = 0

        def get_finite_dict(self, name):
            return cache.DictFiniteCapacity(capacity=1)

        @cache.cached_method(
            hash_function=serializer, ignore_args=ignore_args, factory=cache_factory
        )
        def cached(self, arg, display=True):
            self.counter += 1
            return arg

    obj = CacheTest()

    # test simple caching behavior
    assert obj.cached(1, display=True) == 1
    assert obj.counter == 1
    assert obj.cached(1, display=True) == 1
    assert obj.counter == 1
    assert obj.cached(1, display=False) == 1
    assert obj.counter == 1
    assert obj.cached(2, display=True) == 2
    assert obj.counter == 2
    assert obj.cached(2, display=False) == 2
    assert obj.counter == 2
    assert obj.cached(2, display=False) == 2
    assert obj.counter == 2


def test_cache_clearing():
    """ make sure that memory is freed when cache is cleared """

    class Test:
        """ simple test object with a cache """

        @cache.cached_method()
        def calc(self, n):
            return np.empty(n)

        def clear_cache(self):
            self._cache_methods = {}

        def clear_specific(self):
            self.calc.clear_cache_of_obj(self)

    t = Test()

    mem0 = deep_getsizeof(t)

    for clear_cache in (t.clear_cache, t.clear_specific):
        t.calc(100)
        mem1 = deep_getsizeof(t)
        assert mem1 > mem0
        t.calc(200)
        mem2 = deep_getsizeof(t)
        assert mem2 > mem1
        t.calc(100)
        mem3 = deep_getsizeof(t)
        assert mem3 == mem2

        clear_cache()
        mem4 = deep_getsizeof(t)
        assert mem4 >= mem0
        assert mem1 >= mem4


def test_serialized_dict():
    """ test SerializedDict """
    d = cache.SerializedDict()
    assert len(d) == 0
    d["a"] = 1
    assert len(d) == 1
    assert "a" in d
    assert d["a"] == 1
    assert list(d) == ["a"]
    del d["a"]
    assert len(d) == 0


def test_finite_dict():
    """ test DictFiniteCapacity """
    d = cache.DictFiniteCapacity(capacity=1)
    d["a"] = 1
    assert d["a"] == 1
    d["b"] = 2
    assert d["b"] == 2
    assert "a" not in d

    d1 = cache.DictFiniteCapacity(capacity=1)
    d2 = cache.DictFiniteCapacity(capacity=1)
    d1["a"] = d2["a"] = 1
    assert d1 == d2
    d2 = cache.DictFiniteCapacity(capacity=2)
    d2["a"] = 1
    assert d1 != d2
