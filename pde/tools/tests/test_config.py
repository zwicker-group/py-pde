"""
.. codeauthor:: David Zwicker <dzwicker@seas.harvard.edu>
"""

import pytest

from pde.tools.config import Config


def test_config():
    """ test configuration system """
    c = Config()

    assert c["numba.parallel_threshold"] > 0

    assert "numba.parallel_threshold" in c
    assert any("numba.parallel_threshold" == k for k in c)
    assert any("numba.parallel_threshold" == k and v > 0 for k, v in c.items())
    assert "numba.parallel_threshold" in c.to_dict()


def test_config_modes():
    """ test configuration system running in different modes """
    c = Config(mode="insert")
    assert c["numba.parallel_threshold"] > 0
    c["numba.parallel_threshold"] = 0
    assert c["numba.parallel_threshold"] == 0

    c["new_value"] = "value"
    assert c["new_value"] == "value"

    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="update")
    assert c["numba.parallel_threshold"] > 0
    c["numba.parallel_threshold"] = 0
    assert c["numba.parallel_threshold"] == 0

    with pytest.raises(KeyError):
        c["new_value"] = "value"

    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="locked")
    assert c["numba.parallel_threshold"] > 0
    with pytest.raises(RuntimeError):
        c["numba.parallel_threshold"] = 0

    with pytest.raises(RuntimeError):
        c["new_value"] = "value"

    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="undefined")
    assert c["numba.parallel_threshold"] > 0
    with pytest.raises(ValueError):
        c["numba.parallel_threshold"] = 0


def test_config_special_values():
    """ test typical config objects """
    from pde.tools.numba import NUMBA_PARALLEL

    c = Config(mode="insert")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    c["numba.parallel"] = not NUMBA_PARALLEL
    assert c["numba.parallel"] != NUMBA_PARALLEL

    c = Config(mode="update")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    c["numba.parallel"] = not NUMBA_PARALLEL
    assert c["numba.parallel"] != NUMBA_PARALLEL

    c = Config(mode="locked")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    with pytest.raises(RuntimeError):
        c["numba.parallel"] = not NUMBA_PARALLEL

    c = Config(mode="undefined")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    with pytest.raises(ValueError):
        c["numba.parallel"] = not NUMBA_PARALLEL
