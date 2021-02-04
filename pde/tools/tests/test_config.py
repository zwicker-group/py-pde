"""
.. codeauthor:: David Zwicker <dzwicker@seas.harvard.edu>
"""

import pytest

from pde.tools.config import Config


def test_config():
    """ test typical config objects """
    from pde.tools.numba import NUMBA_PARALLEL

    c = Config(mode="insert")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    c["numba.parallel"] = not NUMBA_PARALLEL
    assert c["numba.parallel"] != NUMBA_PARALLEL

    c["new_value"] = "value"
    assert c["new_value"] == "value"

    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="update")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    c["numba.parallel"] = not NUMBA_PARALLEL
    assert c["numba.parallel"] != NUMBA_PARALLEL

    with pytest.raises(KeyError):
        c["new_value"] = "value"

    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="locked")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    with pytest.raises(RuntimeError):
        c["numba.parallel"] = not NUMBA_PARALLEL

    with pytest.raises(RuntimeError):
        c["new_value"] = "value"

    with pytest.raises(KeyError):
        c["undefined"]

    c = Config(mode="undefined")
    assert c["numba.parallel"] == NUMBA_PARALLEL
    with pytest.raises(ValueError):
        c["numba.parallel"] = not NUMBA_PARALLEL
