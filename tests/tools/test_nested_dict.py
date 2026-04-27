"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import copy

import pytest

from pde.tools.nested_dict import NestedDict


def test_nested_dict_basics():
    """Tests miscellaneous functions."""
    d = NestedDict({"a": {"b": {}}, "c": 1})

    assert isinstance(repr(d), str)

    #     d.pprint()
    # self.assertGreater(len(stream.getvalue()), 0)


def test_nested_dict_getting_data():
    """Tests that are about retrieving data."""
    d = NestedDict({"a": {"b": 1}, "c": 2})
    assert d["a"] == NestedDict({"b": 1})
    assert d["a.b"] == 1
    assert d["c"] == 2

    with pytest.raises(KeyError):
        d["z"]
    with pytest.raises(KeyError):
        d["a.z"]
    with pytest.raises(TypeError):
        d["a.b.z"]
    with pytest.raises(TypeError):
        d["c.z"]


def test_nested_dict_membership():
    """Tests that test membership."""
    d = NestedDict({"a": {"b": 1}, "c": 2})

    assert "a" in d
    assert "a.b" in d
    assert "c" in d

    assert "z" not in d
    assert "a.z" not in d
    assert "c.z" not in d


def test_nested_dict_empty_dict():
    """Tests that are about retrieving data."""
    d = NestedDict({"a": {"b": {}}, "c": 1})
    assert d.to_dict(flatten=True) == {"c": 1}


def test_nested_dict_setting_data():
    """Tests that are about setting data."""
    d = NestedDict({"a": {"b": {}}})

    d["a.c"] = 2
    assert d["a.c"] == 2

    d["e.f"] = 3
    assert d["e.f"] == 3

    assert d.to_dict() == {"a": {"b": {}, "c": 2}, "e": {"f": 3}}
    assert d.to_dict(flatten=True) == {"a.c": 2, "e.f": 3}

    d = NestedDict({"a": {"b": {}}})
    with pytest.raises(TypeError):
        d["a"] = 2
    with pytest.raises(TypeError):
        d["a.b"] = 2

    r = d["f"] = {"1": 2}
    assert r == NestedDict({"1": 2})
    assert d == NestedDict({"a": {"b": {}}, "f": {"1": 2}})

    d = NestedDict({"a": {"b": 1}})
    d.update_recursive({"a.c": 2})
    assert d == NestedDict({"a": {"b": 1, "c": 2}})
    d.update_recursive({"a": {"c": 3}})
    assert d == NestedDict({"a": {"b": 1, "c": 3}})

    # test inplace operation
    d = NestedDict({"a": {"b": 1}})
    d["a.b"] += 1
    assert d == NestedDict({"a": {"b": 2}})
    d["a.b"] *= 2
    assert d == NestedDict({"a": {"b": 4}})


def test_nested_dict_deleting_data():
    """Tests that are about deleting data."""
    d = NestedDict({"a": {"b": 1, "c": 2}, "d": 3})

    with pytest.raises(KeyError):
        del d["g"]

    with pytest.raises(TypeError):
        del d["d.z"]

    del d["d"]
    assert d == NestedDict({"a": {"b": 1, "c": 2}})

    del d["a.c"]
    assert d == NestedDict({"a": {"b": 1}})

    del d["a"]
    assert d == NestedDict()


def test_nested_dict_iterators():
    """Test iterating over the data."""
    d = NestedDict({"a": {"b": 1}, "c": 2})

    assert set(d.keys()) == {"a", "c"}
    assert set(d.keys(flatten=True)) == {"a.b", "c"}

    assert len(list(d.values())) == 2
    assert 2 in d.values()
    assert set(d.values(flatten=True)) == {1, 2}

    assert len(list(d.items())) == 2
    assert ("c", 2) in d.items()
    assert set(d.items(flatten=True)) == {("a.b", 1), ("c", 2)}

    # test some exceptions
    with pytest.raises(TypeError):
        list(NestedDict({1: {2: 3}}).keys(flatten=True))
    with pytest.raises(TypeError):
        list(NestedDict({1: {2: 3}}).items(flatten=True))


def test_nested_dict_conversion():
    """Test the conversion of dictionaries."""
    d = NestedDict({"a": {"b": 1}, "c": 2})

    d2 = d.copy()
    assert d2 == d
    d2["c"] = 3
    assert d2 != d

    d3 = NestedDict(d.to_dict())
    assert d3 == d
    d3 = NestedDict(d.to_dict(flatten=True))
    assert d3 == d

    with pytest.raises(TypeError):
        d = NestedDict({1: {2: 3}})


def test_nested_dict_separtor():
    d = NestedDict({"a": 1})
    assert d == {"a": 1}

    class DashNestedDict(NestedDict):
        sep = "/"

    d = DashNestedDict({"a/b": 1})
    assert d == {"a": {"b": 1}}

    d = DashNestedDict({"a/b/c": 1})
    assert d == {"a": {"b": {"c": 1}}}


def test_nested_dict_copy():
    """Test copies (including copy.copy and copy.deepcopy)"""

    class MockValue:
        def __init__(self, val):
            self.val = val

        def __eq__(self, other):
            if not isinstance(other, MockValue):
                return NotImplemented
            return self.val == other.val

    a = NestedDict({"a": {"b": MockValue(1)}})

    b = a.copy()
    assert a == b
    assert id(a) != id(b)
    assert id(a["a"]) != id(b["a"])
    assert id(a["a"]["b"]) == id(b["a"]["b"])

    b = copy.copy(a)
    assert a == b
    assert id(a) != id(b)
    assert id(a["a"]) == id(b["a"])
    assert id(a["a"]["b"]) == id(b["a"]["b"])

    b = copy.deepcopy(a)
    assert a == b
    assert id(a) != id(b)
    assert id(a["a"]) != id(b["a"])
    assert id(a["a"]["b"]) != id(b["a"]["b"])


def test_nested_dict_update():
    """Test the difference between update and update recursive."""
    d = NestedDict({"a": {"b": 1}})
    with pytest.raises(TypeError):
        d.update({"a": 2})

    d = NestedDict({"a": {"b": 1}})
    d.update_recursive({"a": {"c": 2}})
    assert d.to_dict() == {"a": {"b": 1, "c": 2}}


def test_nested_dict_insert():
    d = NestedDict({"a": None})
    with pytest.raises(TypeError):
        d["a"] = {"b": 1}

    d = NestedDict({"a": {}})
    d["a"] = {"b": 1}
    assert d["a.b"] == 1
    assert d["a"]["b"] == 1


def test_nested_dict_error_paths_and_clear():
    """Test type errors and clear method behavior."""
    d = NestedDict({"a": {"b": 1}})

    with pytest.raises(TypeError):
        d[1]
    with pytest.raises(TypeError):
        d[1] = 2
    with pytest.raises(TypeError):
        d.create_node(1)
    with pytest.raises(TypeError):
        d.update_recursive([("a", 1)])
    with pytest.raises(TypeError):
        d.update([("a", 1)])

    d.clear()
    assert len(d) == 0
    assert d.to_dict() == {}


def test_nested_dict_create_node_nested_path():
    """Test creating nested nodes using create_node."""
    d = NestedDict()

    node = d.create_node("x.y.z")
    assert isinstance(node, NestedDict)
    assert "x.y.z" in d
    assert isinstance(d["x.y.z"], NestedDict)

    d["x.y.z.k"] = 4
    assert d["x.y.z.k"] == 4
