"""Defines utilities for the torch backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import DTypeLike

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


def torch_heaviside(x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
    """Return the Heaviside step function using torch.

    This wraps :func:`torch.heaviside` and ensures that scalar fallback values are
    converted to tensors with a dtype compatible with `x1`.

    Args:
        x1 (:class:`torch.Tensor`):
            Input values at which the Heaviside function is evaluated.
        x2 (:class:`torch.Tensor`, optional):
            Value used where `x1 == 0`. If omitted, `0.5` is used.

    Returns:
        :class:`torch.Tensor`:
            Tensor containing the Heaviside values of `x1`.
    """
    x1_t = torch.tensor(x1)
    if x2 is None:
        return torch.heaviside(x1_t, torch.tensor(0.5, dtype=x1_t.dtype))
    return torch.heaviside(x1_t, torch.tensor(x2, dtype=x1_t.dtype))


def torch_hypot(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Return the Euclidean norm ``sqrt(x1**2 + x2**2)`` using torch.

    This wraps :func:`torch.hypot` and ensures that both inputs are converted to
    tensors before evaluation.

    Args:
        x1 (:class:`torch.Tensor`):
            First input values.
        x2 (:class:`torch.Tensor`):
            Second input values.

    Returns:
        :class:`torch.Tensor`:
            Tensor containing the element-wise hypotenuse of `x1` and `x2`.
    """
    return torch.hypot(torch.tensor(x1), torch.tensor(x2))


SPECIAL_FUNCTIONS_TORCH: dict[str, Callable] = {
    "Heaviside": torch_heaviside,
    "hypot": torch_hypot,
}
