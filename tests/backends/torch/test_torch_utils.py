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

from pde.backends.torch.utils import get_torch_dtype


def test_torch_dtype():
    """Test the `get_torch_dtype` function."""
    assert torch.float64 == get_torch_dtype(np.float64)
    assert torch.float64 == get_torch_dtype("float64")
    assert torch.float32 == get_torch_dtype(np.float32)
