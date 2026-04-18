"""Test general backend selection.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform

import pytest

pytest.importorskip("torch")
if platform.system() == "Windows":
    pytest.skip("Skip torch tests on Windows", allow_module_level=True)

from pde import CahnHilliardPDE, ScalarField, UnitGrid, config, get_backend
from pde.backends import BackendBase, backend_registry
from pde.backends.torch import TorchBackend


@pytest.mark.parametrize(
    "backend", ["torch-cpu", "torch-mps", "torch-cuda"], indirect=True
)
def test_backend_selection(backend, rng):
    """Test whether backends can be easily constructed."""
    # try setting the torch device explicitly by creating a new backend on the fly
    device = str(backend.device)
    try:
        new_backend = TorchBackend(name=f"my-torch-{device}", device=device)
    except RuntimeError:
        pytest.skip(f"Device `{device}` is not available")

    state = ScalarField.random_uniform(UnitGrid([3]), rng=rng)
    # it is important that we use a PDE of sufficient complexity here, so that we test
    # that operators are created for the correct backend.
    eq = CahnHilliardPDE()
    eq.solve(state, t_range=1, backend=new_backend, tracker=None)

    assert eq.diagnostics["solver"]["backend"]["name"] == f"my-torch-{device}"
    assert eq.diagnostics["solver"]["backend"]["device"] == device


def test_backend_configuration_known():
    """Test some aspects of configuration options of known backends."""
    # test different ways to access the config
    assert config["backend.numba.fastmath"]
    assert backend_registry.get_config("numba")["fastmath"]
    assert get_backend("numba").config["fastmath"]
    assert backend_registry.get_config("numba") is get_backend("numba").config

    get_backend("numba").config["fastmath"] = False
    assert not config["backend.numba.fastmath"]
    assert not backend_registry.get_config("numba")["fastmath"]
    assert not get_backend("numba").config["fastmath"]

    config["backend.numba.fastmath"] = True
    assert config["backend.numba.fastmath"]
    assert backend_registry.get_config("numba")["fastmath"]
    assert get_backend("numba").config["fastmath"]


def test_backend_configuration_special():
    """Test some aspects of configuration options with specialized backends."""
    # test different ways to access the config
    assert backend_registry.get_config("torch")["dtype_downcasting"]

    # standard backend must modify global config
    get_backend("torch").config["dtype_downcasting"] = False
    assert not backend_registry.get_config("torch")["dtype_downcasting"]

    # global config must change standard backend
    backend_registry.get_config("torch")["dtype_downcasting"] = True
    assert get_backend("torch").config["dtype_downcasting"]

    # special backend behaves differently
    special_backend = backend_registry.get_backend(
        "torch:cpu", config={"dtype_downcasting": False}
    )
    assert not special_backend.config["dtype_downcasting"]
    assert not get_backend("torch:cpu").config["dtype_downcasting"]
    assert isinstance(special_backend, TorchBackend)
    assert get_backend("torch").config["dtype_downcasting"]
    assert backend_registry.get_config("torch")["dtype_downcasting"]


def test_backend_configuration_new():
    """Test configuration of new backend."""

    class MyBackend(BackendBase): ...

    backend = MyBackend({"option": 1}, name="test_config")
    backend_registry.register_backend(backend, link_config=True)

    assert backend.config["option"] == 1
    assert backend_registry["test_config"].config["option"] == 1
    assert config["backend.test_config.option"] == 1
    assert backend_registry.get_config("test_config")["option"] == 1
