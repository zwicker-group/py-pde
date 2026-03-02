"""Test general backend selection.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform

import pytest

pytest.importorskip("torch")
if platform.system() == "Windows":
    pytest.skip("Skip torch tests on Windows", allow_module_level=True)

from pde import DiffusionPDE, ScalarField, UnitGrid
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
