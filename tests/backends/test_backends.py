"""Test general backend selection.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform

import pytest

pytest.importorskip("torch")
if platform.system() == "Windows":
    pytest.skip("Skip torch tests on Windows", allow_module_level=True)

from pde import (
    CahnHilliardPDE,
    ScalarField,
    UnitGrid,
    backend_registry,
    config,
    get_backend,
)
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
    # it is important that we use a PDE of sufficient complexity here, so that we test
    # that operators are created for the correct backend.
    eq = CahnHilliardPDE()
    eq.solve(state, t_range=1, backend=backend)

    assert eq.diagnostics["solver"]["backend"]["name"] == f"my-torch-{device}"
    assert eq.diagnostics["solver"]["backend"]["device"] == device


def test_backend_configuration():
    """Test some aspects of configuration options."""
    # test modification of known backend
    assert config["backend.numba.fastmath"]
    assert backend_registry.get_config("numba")["fastmath"]
    assert get_backend("numba").config["fastmath"]

    get_backend("numba").config["fastmath"] = False
    assert not config["backend.numba.fastmath"]
    assert not backend_registry.get_config("numba")["fastmath"]
    assert not get_backend("numba").config["fastmath"]

    config["backend.numba.fastmath"] = True
    assert config["backend.numba.fastmath"]
    assert backend_registry.get_config("numba")["fastmath"]
    assert get_backend("numba").config["fastmath"]

    # test configuration of new backend
    class MyBackend(BackendBase): ...

    backend = MyBackend({"option": 1}, name="test_config")
    backend_registry.add(backend)

    assert backend.config["option"] == 1
    assert backend_registry["test_config"].config["option"] == 1
    assert config["backend.test_config.option"] == 1
    assert backend_registry.get_config("test_config")["option"] == 1
