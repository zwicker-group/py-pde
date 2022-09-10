"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde.tools.config import Config, environment


def test_environment():
    """test the environment function"""
    assert isinstance(environment(), dict)


def test_config():
    """test configuration system"""
    c = Config()

    assert c["numba.multithreading_threshold"] > 0

    assert "numba.multithreading_threshold" in c
    assert any("numba.multithreading_threshold" == k for k in c)
    assert any("numba.multithreading_threshold" == k and v > 0 for k, v in c.items())
    assert "numba.multithreading_threshold" in c.to_dict()
    assert isinstance(repr(c), str)


def test_config_modes():
    """test configuration system running in different modes"""
    c = Config(mode="insert")
    assert c["numba.multithreading_threshold"] > 0
    c["numba.multithreading_threshold"] = 0
    assert c["numba.multithreading_threshold"] == 0
    c["new_value"] = "value"
    assert c["new_value"] == "value"
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
    with pytest.raises(RuntimeError):
        del c["numba.multithreading_threshold"]
    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="locked")
    assert c["numba.multithreading_threshold"] > 0
    with pytest.raises(RuntimeError):
        c["numba.multithreading_threshold"] = 0
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
    with pytest.raises(RuntimeError):
        del c["numba.multithreading_threshold"]

    c = Config({"new_value": "value"}, mode="locked")
    assert c["new_value"] == "value"


def test_config_contexts():
    """test context manager temporarily changing configuration"""
    c = Config()

    assert c["numba.multithreading_threshold"] > 0
    with c({"numba.multithreading_threshold": 0}):
        assert c["numba.multithreading_threshold"] == 0
        with c({"numba.multithreading_threshold": 1}):
            assert c["numba.multithreading_threshold"] == 1
        assert c["numba.multithreading_threshold"] == 0

    assert c["numba.multithreading_threshold"] > 0
