"""Defines utilities for the pytorch backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor

AnyDType = str | np.dtype[np.generic] | torch.dtype

NUMPY_TO_TORCH_DTYPE: dict[np.dtype[np.generic], torch.dtype] = {
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
NUMPY_TO_TORCH_DTYPE = {np.dtype(k): v for k, v in NUMPY_TO_TORCH_DTYPE.items()}


def get_torch_dtype(dtype: AnyDType) -> torch.dtype:
    """Convert dtype to torch dtype.

    Args:
        dtype:
            dtype which could be a python type, a numpy dtype, or already a torch dtype

    Returns:
        :class:`torch.dtype`:
            A proper dtype for torch
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype = np.dtype(dtype)
    return NUMPY_TO_TORCH_DTYPE[dtype]


class TorchOperatorType(Protocol):
    """An operator that acts on an array."""

    def __call__(
        self,
        arr: Tensor,
        out: Tensor | None = None,
        args: dict[str, Any] | None = None,
    ) -> Tensor:
        """Evaluate the operator."""
