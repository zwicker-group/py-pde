"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import platform

import numpy as np
import pytest

pytest.importorskip("torch")
if platform.system() == "Windows":
    pytest.skip("Skip torch tests on Windows", allow_module_level=True)

import torch


@pytest.mark.parametrize(
    "backend", ["torch-cpu", "torch-mps", "torch-cuda"], indirect=True
)
def test_torch_dtype(backend):
    """Test the `get_torch_dtype` function."""
    assert torch.float32 == backend.get_torch_dtype(np.float32)
    assert torch.float32 == backend.get_torch_dtype("float32")

    if backend.name == "torch-mps":
        assert torch.float32 == backend.get_torch_dtype(np.float64)
    else:
        assert torch.float64 == backend.get_torch_dtype(np.float64)
