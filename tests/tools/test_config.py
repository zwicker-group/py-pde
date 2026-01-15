"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde.tools.config import (
    Config,
    GlobalConfig,
    Parameter,
    environment,
    packages_from_requirements,
)


def test_environment():
    """Test the environment function."""
    env = environment()
    assert isinstance(env, dict)
    assert isinstance(env["config"], dict)
    assert isinstance(env["config"]["default_backend"], str)


def test_config_basic():
    """Test basic configuration system."""
    c = GlobalConfig([Parameter("key", 1)])
    assert c["key"] > 0
    assert isinstance(c.to_dict(ret_values=False)["key"], Parameter)
    assert c.to_dict(ret_values=True)["key"] == 1
    c["key"] = 0
    assert c["key"] == 0


def test_config_backends():
    """Test configuration system."""
    # initialize config without anything to test whether it refers to backends
    c = GlobalConfig()

    assert c["backend.numba.multithreading_threshold"] > 0

    assert "backend.numba.multithreading_threshold" in c
    assert any(k == "backend.numba.multithreading_threshold" for k in c)
    assert any(
        k == "backend.numba.multithreading_threshold" and v > 0 for k, v in c.items()
    )
    assert "backend.numba.multithreading_threshold" in c.to_dict()
    assert isinstance(repr(c), str)


def test_config_modes():
    """Test configuration system running in different modes."""
    c = GlobalConfig({"key": 3}, mode="insert")
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

    c = GlobalConfig({"key": 3}, mode="update")
    assert c["key"] > 0
    c["key"] = 0

    with pytest.raises(KeyError):
        c["new_value"] = "value"
    with pytest.raises(KeyError):
        c.update({"new_value": "value"})
    with pytest.raises(RuntimeError):
        del c["backend.numba.multithreading_threshold"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = GlobalConfig({"key": 3}, mode="locked")
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

    c = GlobalConfig({"key": 3}, mode="undefined")
    assert c["key"] > 0
    with pytest.raises(ValueError):
        c["key"] = 0
    with pytest.raises(ValueError):
        c.update({"key": 0})
    with pytest.raises(RuntimeError):
        del c["key"]

    c = GlobalConfig({"new_value": "value"}, mode="locked")
    assert c["new_value"] == "value"


def test_config_contexts():
    """Test context manager temporarily changing configuration."""
    c = Config({"key": 3})

    assert c["key"] == 3
    with c({"key": 0}):
        assert c["key"] == 0
        with c({"key": 1}):
            assert c["key"] == 1
        assert c["key"] == 0

    assert c["key"] == 3


def test_packages_from_requirements():
    """Test the packages_from_requirements function."""
    results = packages_from_requirements("file_not_existing")
    assert len(results) == 1
    assert "Could not open" in results[0]
    assert "file_not_existing" in results[0]
