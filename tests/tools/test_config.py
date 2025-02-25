"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde.tools.config import Config, environment, packages_from_requirements


def test_environment():
    """Test the environment function."""
    assert isinstance(environment(), dict)


def test_config():
    """Test configuration system."""
    c = Config()

    assert c["numba.multithreading_threshold"] > 0

    assert "numba.multithreading_threshold" in c
    assert any(k == "numba.multithreading_threshold" for k in c)
    assert any(k == "numba.multithreading_threshold" and v > 0 for k, v in c.items())
    assert "numba.multithreading_threshold" in c.to_dict()
    assert isinstance(repr(c), str)


def test_config_modes():
    """Test configuration system running in different modes."""
    c = Config(mode="insert")
    assert c["numba.multithreading_threshold"] > 0
    c["numba.multithreading_threshold"] = 0
    assert c["numba.multithreading_threshold"] == 0
    c["new_value"] = "value"
    assert c["new_value"] == "value"
    c.update({"new_value2": "value2"})
    assert c["new_value2"] == "value2"
    del c["new_value"]
    with pytest.raises(KeyError):
        c["new_value"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="update")
    assert c["numba.multithreading_threshold"] > 0
    c["numba.multithreading_threshold"] = 0

    with pytest.raises(KeyError):
        c["new_value"] = "value"
    with pytest.raises(KeyError):
        c.update({"new_value": "value"})
    with pytest.raises(RuntimeError):
        del c["numba.multithreading_threshold"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="locked")
    assert c["numba.multithreading_threshold"] > 0
    with pytest.raises(RuntimeError):
        c["numba.multithreading_threshold"] = 0
    with pytest.raises(RuntimeError):
        c.update({"numba.multithreading_threshold": 0})
    with pytest.raises(RuntimeError):
        c["new_value"] = "value"
    with pytest.raises(RuntimeError):
        del c["numba.multithreading_threshold"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="undefined")
    assert c["numba.multithreading_threshold"] > 0
    with pytest.raises(ValueError):
        c["numba.multithreading_threshold"] = 0
    with pytest.raises(ValueError):
        c.update({"numba.multithreading_threshold": 0})
    with pytest.raises(RuntimeError):
        del c["numba.multithreading_threshold"]

    c = Config({"new_value": "value"}, mode="locked")
    assert c["new_value"] == "value"


def test_config_contexts():
    """Test context manager temporarily changing configuration."""
    c = Config()

    assert c["numba.multithreading_threshold"] > 0
    with c({"numba.multithreading_threshold": 0}):
        assert c["numba.multithreading_threshold"] == 0
        with c({"numba.multithreading_threshold": 1}):
            assert c["numba.multithreading_threshold"] == 1
        assert c["numba.multithreading_threshold"] == 0

    assert c["numba.multithreading_threshold"] > 0


def test_config_special_values():
    """Test configuration system running in different modes."""
    c = Config()
    c["numba.multithreading"] = True
    assert c["numba.multithreading"] == "always"
    assert c.use_multithreading()
    c["numba.multithreading"] = False
    assert c["numba.multithreading"] == "never"
    assert not c.use_multithreading()


def test_packages_from_requirements():
    """Test the packages_from_requirements function."""
    results = packages_from_requirements("file_not_existing")
    assert len(results) == 1
    assert "Could not open" in results[0]
    assert "file_not_existing" in results[0]
