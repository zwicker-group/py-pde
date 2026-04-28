"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import warnings

import pytest

from pde.backends import backend_registry
from pde.tools import config as config_module
from pde.tools.config import (
    _DEFAULT,
    _OMITTED,
    AccessError,
    Config,
    ConfigMode,
    Modes,
    Parameter,
    _ConfigDataDict,
    config,
    environment,
    packages_from_requirements,
)


def test_config_basic():
    """Test basic configuration system."""
    c = Config({"key": Parameter(1)})
    assert c["key"] == 1
    assert isinstance(c.to_dict(values=False)["key"], Parameter)
    assert c.to_dict(values=True)["key"] == 1
    c["key"] = 0
    assert c["key"] == 0
    assert c.to_dict(values=False)["key"].value == 0


def test_parameter_current_value():
    """Test that parameters store a current value separately from the default."""
    p = Parameter(1)
    assert p.default_value == 1
    assert p.value == 1
    assert p.convert() == 1

    p = Parameter(default_value=1, value=3)
    assert p.default_value == 1
    assert p.value == 3
    assert p.convert() == 3

    p.reset()
    assert p.value == 1
    assert p.convert() == 1


def test_config_backends():
    """Test configuration system."""
    # the package-level config is populated with backend defaults during import
    c = config.copy()

    assert c["backend.backend.numba.multithreading_threshold"] > 0

    assert "backend.backend.numba.multithreading_threshold" in c
    assert set(c.keys(flatten=True)) >= {"backend.backend.numba.multithreading_threshold"}
    assert any(
        k == "backend.backend.numba.multithreading_threshold" and v > 0
        for k, v in c.to_dict(flatten=True, values=True).items()
    )
    assert "backend.backend.numba.multithreading_threshold" in c.to_dict(flatten=True)
    assert isinstance(repr(c), str)


def test_config_nested_structure():
    """Test some aspects of the nested structure of configurations."""
    c = Config(mode="insert")
    c["a.b.c"] = 1
    c["a.b.d"] = 2
    assert len(c) == 1
    assert len(c.to_dict(flatten=True)) == 2


def test_config_nested_dict_interface():
    """Test that Config exposes the NestedDict traversal helpers."""
    c = Config({"a": {"b": Parameter(1)}})

    assert c["a.b"] == 1
    assert set(c.keys(flatten=True)) == {"a.b"}
    assert list(c.items(flatten=True)) == [("a.b", c._get_raw_item("a.b"))]
    assert c.to_dict(flatten=True, values=True) == {"a.b": 1}


def test_config_updates_parameter_value():
    """Test that assigning to a config key updates the parameter value."""
    c = Config({"key": Parameter(1.2, cls=int)})
    assert c["key"] == 1
    c["key"] = 4
    assert c["key"] == 4
    assert c.to_dict(values=False)["key"] != Parameter(4)  # different default value
    assert c.to_dict(values=True) == {"key": 4}


def test_config_auto_creates_parameter_for_missing_leaf():
    """Test that setting a missing leaf creates a Parameter instance."""
    c = Config({}, mode="insert")

    c["branch.leaf"] = 12

    raw_leaf = c._get_raw_item("branch.leaf")
    assert isinstance(raw_leaf, Parameter)
    assert raw_leaf.default_value == 12
    assert raw_leaf.value == 12
    assert c["branch.leaf"] == 12


def test_config_tree_contains_only_configs_and_parameters():
    """Test that internal config data only consists of Config and Parameter values."""
    c = Config({"a": {"b": 1}, "c": Parameter(2)}, mode="insert")

    assert isinstance(c._get_raw_item("a"), Config)
    assert isinstance(c._get_raw_item("a.b"), Parameter)
    assert isinstance(c._get_raw_item("c"), Parameter)


def test_config_requires_string_keys():
    """Test that Config enforces string keys."""
    c = Config({}, mode="insert")

    with pytest.raises(TypeError):
        c[1] = 2


def test_config_modes_init():
    """Test the initialization of the configuration modes."""
    m = ConfigMode.from_str("insert")
    assert m.node == Modes.INSERT
    assert m.leaf == Modes.INSERT
    assert not m.delete
    m = ConfigMode.from_str("update")
    assert m.node == Modes.UPDATE
    assert m.leaf == Modes.UPDATE
    assert not m.delete
    m = ConfigMode.from_str("readonly")
    assert m.node == Modes.READONLY
    assert m.leaf == Modes.READONLY
    assert not m.delete


def test_config_modes():
    """Test configuration system running in different modes."""
    c = Config({"key": 3}, mode=ConfigMode(node="insert", leaf="insert", delete=True))
    assert c.mode.node == Modes.INSERT
    assert c.mode.leaf == Modes.INSERT
    assert c.mode.delete
    assert c["key"] > 0
    c["key"] = 0
    assert c["key"] == 0
    c["new_value"] = "value"
    assert c["new_value"] == "value"
    c.update({"new_value2": "value2"})
    assert c["new_value2"] == "value2"
    del c["new_value"]
    with pytest.raises(KeyError):
        c["new_value"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config({"key": 3}, mode="update")
    assert c.mode.node == Modes.UPDATE
    assert c.mode.leaf == Modes.UPDATE
    assert not c.mode.delete
    assert c["key"] > 0
    c["key"] = 0

    with pytest.raises(AccessError):
        c["new_value"] = "value"
    with pytest.raises(AccessError):
        c.update({"new_value": "value"})
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config({"key": 3}, mode="readonly")
    assert c["key"] > 0
    with pytest.raises(RuntimeError):
        c["key"] = 0
    with pytest.raises(RuntimeError):
        c.update({"key": 0})
    with pytest.raises(RuntimeError):
        c["new_value"] = "value"
    with pytest.raises(RuntimeError):
        del c["key"]
    with pytest.raises(KeyError):
        c["undefined"]

    with pytest.raises(ValueError):
        c = Config({"key": 3}, mode="undefined")

    c = Config({"new_value": "value"}, mode="readonly")
    assert c["new_value"] == "value"


def test_config_consistency():
    """Test whether all modes are the same."""
    mode = config._mode
    for value in config.values(flatten=True):
        if isinstance(value, Config):
            assert value._mode == mode

    for name in backend_registry._packages.keys() | backend_registry._classes.keys():
        try:
            backend = backend_registry.get_backend(name)
        except ImportError:
            pass
        else:
            assert backend.config is config["backend"][name]


def test_config_contexts():
    """Test context manager temporarily changing configuration."""
    c = Config({"key": 3}, mode="insert")

    assert c["key"] == 3
    with c({"key": 0}):
        assert c["key"] == 0
        c["add"] = 3
        with c({"key": 1}):
            assert c["key"] == 1
            assert c["add"] == 3
        assert c["key"] == 0
        assert c["add"] == 3

    assert c.to_dict(values=True) == {"key": 3}


def test_config_mode_context_manager():
    """Test that ConfigMode supports nested temporary mode changes."""
    c = Config({"a": 1}, mode="update")
    with pytest.raises(AccessError):
        c["b"] = 2
    with c.changed_mode(node="insert", leaf="insert"):
        c["b"] = 2
        with (
            c.changed_mode(node="readonly", leaf="readonly"),
            pytest.raises(AccessError),
        ):
            c["b"] = 3
        c["b"] = 4
    assert c.mode.node == Modes.UPDATE
    assert c.mode.leaf == Modes.UPDATE
    assert c.to_dict(values=True) == {"a": 1, "b": 4}


def test_config_contextmanager1():
    """Check whether the contextmanager preserves objects."""
    c = config.copy()
    assert set(c.keys(flatten=True)) == set(config.keys(flatten=True))

    ids = {k: id(v) for k, v in c.items(flatten=True)}
    default_backend = c["default_backend"]

    def check_items():
        assert set(c.keys(flatten=True)) == set(config.keys(flatten=True))
        items = dict(c.items(flatten=True))
        for i, id_val in ids.items():
            assert id_val == id(items[i]), f"`{i}` id was changed"

    with c({"default_backend": "none"}):
        # ids should be unchanged
        check_items()

        with c({"default_backend": "nested"}):
            check_items()
            assert c["default_backend"] == "nested"

        check_items()
        assert c["default_backend"] == "none"

    check_items()
    assert c["default_backend"] == default_backend


def test_environment():
    """Test the environment function."""
    env = environment()
    assert isinstance(env, dict)
    assert isinstance(env["config"], dict)
    assert isinstance(env["config"]["default_backend"], str)
    assert isinstance(env["config"]["backend.numba.debug"], bool)


def test_packages_from_requirements():
    """Test the packages_from_requirements function."""
    results = packages_from_requirements("file_not_existing")
    assert len(results) == 1
    assert "Could not open" in results[0]
    assert "file_not_existing" in results[0]


def test_parameter_internal_branches():
    """Test less common branches of parameter initialization and conversion."""
    p = Parameter(_DEFAULT)
    assert p.default_value is None
    assert p.cls is object

    p = Parameter("x")
    assert p.cls is str

    p = Parameter("x", cls=int)
    with pytest.raises(ValueError):
        p.convert()

    p = Parameter(1)
    p.default_value = _OMITTED
    with pytest.raises(RuntimeError):
        p.reset()


def test_config_mode_setstate_and_invalid_allow():
    """Test mode state mutation and invalid category handling."""
    mode = ConfigMode.from_str("insert")
    mode._setstate(node="readonly")
    assert mode.node is Modes.READONLY
    assert mode.leaf is Modes.INSERT

    dct = _ConfigDataDict(mode)
    with pytest.raises(ValueError):
        dct._allow("invalid", {Modes.INSERT})


def test_config_data_dict_guards():
    """Test access guard branches in `_ConfigDataDict`."""
    mode = ConfigMode.from_str("update")
    dct = _ConfigDataDict(mode)

    with pytest.raises(AccessError):
        dct["branch"] = {}

    with pytest.raises(AccessError):
        dct["leaf"] = 1

    mode = ConfigMode.from_str("insert")
    dct = _ConfigDataDict(mode)
    dct["leaf"] = 1
    dct["leaf"] = Parameter(2)
    assert isinstance(dct["leaf"], Parameter)

    dct["node"] = {}
    with pytest.raises(RuntimeError):
        dct["node"] = {}

    dict.__setitem__(dct, "broken", 1)
    with pytest.raises(TypeError):
        _ = dct["broken"]

    with pytest.raises(AccessError):
        dct.clear()


def test_config_error_and_setter_branches():
    """Test branches for mode validation and custom mode setter."""
    with pytest.raises(TypeError):
        Config(mode=1)

    with pytest.raises(TypeError):
        Config(items=[Parameter(1)], mode="insert")

    c = Config({"a": 1}, mode="insert")
    c.mode = ConfigMode.from_str("readonly")
    assert c.mode.node is Modes.READONLY

    dict.__setitem__(c.data, "bad", 1)
    with pytest.raises(TypeError):
        _ = c["bad"]

    with pytest.raises(TypeError):
        c.replace_recursive([("a", 1)])


def test_config_copy_and_context_branches():
    """Test copy fallback branch and contextmanager branch without values dict."""
    c = Config({"a": 1}, mode="insert")
    dict.__setitem__(c.data, "raw", 5)
    c2 = c.copy()
    assert isinstance(c2.data["raw"], Parameter)
    assert c2["raw"] == 5

    c = Config({"a": 1}, mode="insert")
    with c(a=2):
        assert c["a"] == 2
    assert c["a"] == 1


def test_utility_function_branches(monkeypatch):
    """Test utility helper branches for parsing, warnings, and ffmpeg detection."""
    assert config_module.parse_version_str("1.beta.3") == [1, 3]

    def raise_import_error(_name):
        raise ImportError

    monkeypatch.setattr(config_module.importlib, "import_module", raise_import_error)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        config_module.check_package_version("some_package", "1.0")
    assert any("required for py-pde" in str(w.message) for w in rec)

    class Module:
        __version__ = "0.1.0"

    monkeypatch.setattr(config_module.importlib, "import_module", lambda _n: Module)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        config_module.check_package_version("some_package", "2.0")
    assert any("installed" in str(w.message) for w in rec)

    monkeypatch.setattr(config_module.sp, "check_output", lambda *_a, **_k: b"dummy")
    assert config_module.get_ffmpeg_version() is None

    def raise_any(*_a, **_k):
        raise RuntimeError

    monkeypatch.setattr(config_module.sp, "check_output", raise_any)
    assert config_module.get_ffmpeg_version() is None
