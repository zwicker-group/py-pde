"""Defines utilities for the pytorch backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

NUMPY_TO_TORCH_DTYPE = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


class TorchOperatorType(Protocol):
    """An operator that acts on an array."""

    def __call__(
        self,
        arr: Tensor,
        out: Tensor | None = None,
        args: dict[str, Any] | None = None,
    ) -> Tensor:
        """Evaluate the operator."""
