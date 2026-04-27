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
from pde.tools.misc import module_available


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
    # with config():  # ensure config will be reset after this
    # test different ways to access the config
    assert config["backend.numba.fastmath"]
    assert get_backend("numba").config["fastmath"]
    assert config["backend"]["numba"] is get_backend("numba").config
    assert config["backend.numba"] is get_backend("numba").config

    # # standard backend must be modifiable and change global config
    get_backend("numba").config["fastmath"] = False
    assert not config["backend.numba.fastmath"]
    assert not get_backend("numba").config["fastmath"]

    # test different ways to access the config
    assert config["backend.torch.dtype_downcasting"]
    assert get_backend("torch").config["dtype_downcasting"]

    # standard backend must be modifiable and change global config
    get_backend("torch").config["dtype_downcasting"] = False
    assert not config["backend.torch.dtype_downcasting"]
    assert not get_backend("torch").config["dtype_downcasting"]

    # check whether values were reset
    get_backend("numba").config["fastmath"] = True
    get_backend("torch").config["dtype_downcasting"] = True


def test_backend_configuration_new():
    """Test configuration of new backend."""

    class MyBackend(BackendBase): ...

    backend = MyBackend({"option": 1}, name="test_config")
    backend_registry.register_backend(backend)
    assert backend.config["option"] == 1
    assert get_backend("test_config").config["option"] == 1


@pytest.mark.skipif(not module_available("torch"), reason="requires `torch` module")
def test_backend_configuration_subcase():
    """Test configuration of sub-backends."""
    b0 = backend_registry.get_backend("torch")
    b1 = backend_registry.get_backend("torch:cpu")
    assert config["backend.torch.dtype_downcasting"]
    assert b0.config["dtype_downcasting"]
    assert b1.config["dtype_downcasting"]

    b1.config["dtype_downcasting"] = False
    assert config["backend.torch.dtype_downcasting"]
    assert b0.config["dtype_downcasting"]
    assert not b1.config["dtype_downcasting"]

    b0.config["dtype_downcasting"] = False
    assert not config["backend.torch.dtype_downcasting"]
    assert not b0.config["dtype_downcasting"]
    assert not b1.config["dtype_downcasting"]

    b0.config["dtype_downcasting"] = True
    b1.config["dtype_downcasting"] = True
    assert config["backend.torch.dtype_downcasting"]
    assert b0.config["dtype_downcasting"]
    assert b1.config["dtype_downcasting"]
