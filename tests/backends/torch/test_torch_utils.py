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

from pde.backends.torch import torch_backend


def test_torch_dtype():
    """Test the `get_torch_dtype` function."""
    assert torch.float32 == torch_backend.get_torch_dtype(np.float32)
    assert torch.float32 == torch_backend.get_torch_dtype("float32")
