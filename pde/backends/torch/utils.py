"""Defines utilities for the torch backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import DTypeLike
from torch import Tensor

AnyDType = DTypeLike | torch.dtype

NUMPY_TO_TORCH_DTYPE: dict[DTypeLike, torch.dtype] = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.double: torch.double,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
# also define inverse mapping to proper numpy dtypes
TORCH_TO_NUMPY_DTYPE = {v: np.dtype(k) for k, v in NUMPY_TO_TORCH_DTYPE.items()}
# add the proper numpy dtype as an alternative
NUMPY_TO_TORCH_DTYPE |= {np.dtype(k): v for k, v in NUMPY_TO_TORCH_DTYPE.items()}


class TorchDifferentialOperatorType(Protocol):
    """An operator that acts on an array."""

    def __call__(
        self,
        arr: Tensor,
        out: Tensor | None = None,
        args: dict[str, Any] | None = None,
    ) -> Tensor:
        """Evaluate the operator."""


class TorchOperatorBase(torch.nn.Module):
    """Base class for operators implemented in torch."""

    def __init__(self, *, dtype: DTypeLike):
        """Initialize the torch operator.

        Args:
            dtype:
                The data type of the field using the numpy convention
        """
        super().__init__()
        self.dtype = np.dtype(dtype)

    def register_array(self, name: str, arr: np.ndarray | torch.Tensor) -> None:
        """Register an array as a buffer in the torch module.

        Args:
            name (str):
                The name under which the buffer is registered
            arr (:class:`numpy.ndarray` or :class:`torch.Tensor`):
                The array to register. If a numpy array is provided, it will be
                converted to a torch tensor with the appropriate dtype.
        """
        if isinstance(arr, np.ndarray):
            tensor = torch.from_numpy(np.asarray(arr, dtype=self.dtype))
        elif isinstance(arr, torch.Tensor):
            tensor = arr
        else:
            raise TypeError

        self.register_buffer(name, tensor)
