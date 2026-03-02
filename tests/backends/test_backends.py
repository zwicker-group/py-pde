"""Test general backend selection.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform

import pytest

pytest.importorskip("torch")
if platform.system() == "Windows":
    pytest.skip("Skip torch tests on Windows", allow_module_level=True)

from pde import DiffusionPDE, ScalarField, UnitGrid, config
from pde.backends import backends
from pde.backends.base import BackendBase
from pde.backends.torch import TorchBackend


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_backend_selection(device, rng):
    """Test whether backends can be easily constructed."""
    try:
        backend = TorchBackend(name=f"my-torch-{device}", device=device)
    except RuntimeError:
        pytest.skip(f"Device `{device}` is not available")

    state = ScalarField.random_uniform(UnitGrid([3]), rng=rng)
    eq = DiffusionPDE()
    eq.solve(state, t_range=1, backend=backend)

    assert eq.diagnostics["solver"]["backend"]["name"] == f"my-torch-{device}"
    assert eq.diagnostics["solver"]["backend"]["device"] == device


def test_backend_configuration():
    """Test some aspects of configuration options."""
    # test modification of known backend
    assert config["backend.numba.fastmath"]
    assert backends.get_config("numba")["fastmath"]
    assert backends["numba"].config["fastmath"]

    backends["numba"].config["fastmath"] = False
    assert not config["backend.numba.fastmath"]
    assert not backends.get_config("numba")["fastmath"]
    assert not backends["numba"].config["fastmath"]

    config["backend.numba.fastmath"] = True
    assert config["backend.numba.fastmath"]
    assert backends.get_config("numba")["fastmath"]
    assert backends["numba"].config["fastmath"]

    # test configuration of new backend
    class MyBackend(BackendBase): ...

    backend = MyBackend({"option": 1}, name="test_config")
    backends.add(backend)

    assert backend.config["option"] == 1
    assert backends["test_config"].config["option"] == 1
    assert config["backend.test_config.option"] == 1
    assert backends.get_config("test_config")["option"] == 1
